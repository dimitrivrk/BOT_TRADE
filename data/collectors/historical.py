"""
Collecteur de données historiques OHLCV via CCXT.
Supporte le téléchargement en masse avec pagination automatique.
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Optional
import yaml
from pathlib import Path

from data.storage.database import MarketDatabase
from utils.logger import setup_logger

logger = setup_logger("data.collectors.historical")


class HistoricalDataCollector:
    """
    Télécharge et met en cache les données historiques OHLCV.
    Gère la pagination, le rate limiting et la reprise incrémentale.
    """

    # Mapping timeframe -> millisecondes
    TF_MS = {
        "1m":   60_000,
        "3m":   180_000,
        "5m":   300_000,
        "15m":  900_000,
        "30m":  1_800_000,
        "1h":   3_600_000,
        "2h":   7_200_000,
        "4h":   14_400_000,
        "6h":   21_600_000,
        "12h":  43_200_000,
        "1d":   86_400_000,
        "3d":   259_200_000,
        "1w":   604_800_000,
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db = MarketDatabase(config_path)
        self.exchange = self._init_exchange()
        self.historical_days = self.config["data"]["historical_days"]

    def _init_exchange(self) -> ccxt.Exchange:
        """
        Initialise la connexion à l'exchange en lecture seule (mainnet).
        Les données historiques publiques ne nécessitent pas d'authentification
        et le testnet n'a pas l'historique complet — on utilise toujours le mainnet ici.
        """
        exc_config = self.config["exchange"]
        exchange_name = exc_config["name"]

        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            "enableRateLimit": exc_config.get("rate_limit", True),
            "timeout": exc_config.get("timeout", 30000),
            # Pas de sandbox ici : données publiques mainnet uniquement
        })

        logger.info(f"Exchange historique initialisé : {exchange_name} (mainnet, lecture seule)")
        return exchange

    def fetch_all_pairs(
        self,
        pairs: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        days: Optional[int] = None,
    ):
        """
        Télécharge les données pour toutes les paires et timeframes configurées.
        Reprend là où il s'était arrêté (reprise incrémentale).
        """
        pairs = pairs or self.config["trading"]["pairs"]
        tf_config = self.config["trading"]["timeframes"]
        timeframes = timeframes or [
            tf_config["primary"],
            tf_config["higher"],
            tf_config["lower"],
        ]
        days = days or self.historical_days

        logger.info(f"Téléchargement données historiques : {len(pairs)} paires × {len(timeframes)} TF × {days} jours")

        for symbol in pairs:
            for timeframe in timeframes:
                try:
                    self.fetch_pair(symbol, timeframe, days)
                    time.sleep(0.5)  # respecter le rate limit
                except Exception as e:
                    logger.error(f"Erreur {symbol}/{timeframe} : {e}")
                    continue

        logger.info("Téléchargement terminé !")

    def fetch_pair(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365,
        force_full: bool = False,
        since_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Télécharge les OHLCV pour une paire/timeframe.
        Reprend depuis le dernier point si déjà en base,
        sauf si since_date est fourni (priorité absolue).
        """
        tf_ms = self.TF_MS.get(timeframe, 3_600_000)
        now_ms = int(time.time() * 1000)

        # Si une date précise est demandée, elle a la priorité absolue
        if since_date:
            since_ms = int(pd.Timestamp(since_date).timestamp() * 1000)
            logger.info(f"{symbol}/{timeframe} : téléchargement depuis {since_date}")
        else:
            # Reprise incrémentale seulement si last_ts est dans le passé raisonnable
            last_ts = self.db.get_last_timestamp(symbol, timeframe)
            target_start_ms = now_ms - (days * 24 * 3600 * 1000)

            if last_ts and not force_full and last_ts >= target_start_ms:
                # DB a deja les donnees recentes -> telecharger seulement les nouvelles bougies
                since_ms = last_ts + tf_ms
                nb_missing = max(0, (now_ms - since_ms) // tf_ms)
                last_dt = pd.Timestamp(last_ts, unit='ms').strftime('%Y-%m-%d %H:%M UTC')
                logger.info(f"{symbol}/{timeframe} : DB a jour jusqu'au {last_dt} | {nb_missing} bougies manquantes")
            else:
                since_ms = target_start_ms
                logger.info(f"{symbol}/{timeframe} : telechargement complet depuis {pd.Timestamp(since_ms, unit='ms')}")

        all_candles = []
        batch_size = 1000  # max Binance

        while since_ms < now_ms - tf_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe,
                    since=since_ms,
                    limit=batch_size,
                )
                if not candles:
                    break

                all_candles.extend(candles)
                since_ms = candles[-1][0] + tf_ms

                logger.debug(
                    f"{symbol}/{timeframe} : {len(all_candles)} bougies "
                    f"(dernière: {pd.Timestamp(candles[-1][0], unit='ms')})"
                )

                if len(candles) < batch_size:
                    break

                time.sleep(self.exchange.rateLimit / 1000)

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limit atteint, pause 60s...")
                time.sleep(60)
            except ccxt.NetworkError as e:
                logger.error(f"Erreur réseau : {e}, retry dans 10s")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Erreur inattendue : {e}")
                break

        if not all_candles:
            logger.info(f"{symbol}/{timeframe} : deja a jour, rien a telecharger")
            return pd.DataFrame()

        df = self._candles_to_df(all_candles)
        self.db.save_ohlcv(df, symbol, timeframe)
        logger.info(f"✓ {symbol}/{timeframe} : {len(df)} bougies sauvegardées")
        return df

    def _candles_to_df(self, candles: list) -> pd.DataFrame:
        """Convertit les candles CCXT en DataFrame pandas."""
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        return df

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Charge les données depuis la base de données, télécharge si nécessaire."""
        df = self.db.load_ohlcv(symbol, timeframe, start=start, end=end)

        if df.empty:
            logger.warning(f"Pas de données en base pour {symbol}/{timeframe}, téléchargement...")
            # Passer la date de début pour télécharger la bonne période
            self.fetch_pair(symbol, timeframe, since_date=start)
            # Recharger depuis la DB après téléchargement
            df = self.db.load_ohlcv(symbol, timeframe, start=start, end=end)

        logger.info(f"Données chargées {symbol}/{timeframe} : {len(df)} bougies")
        return df
