"""
Collecteur de données en temps réel via polling REST Binance.
Remplace WebSocket (CCXT Pro) par polling pour compatibilité maximale.
"""

import asyncio
import ccxt
import pandas as pd
from typing import Dict, List, Callable, Optional
from collections import defaultdict, deque
import yaml
import os
import time
from dotenv import load_dotenv

from data.storage.database import MarketDatabase
from utils.logger import setup_logger

load_dotenv("config/.env")
logger = setup_logger("data.collectors.market_data")

# Intervalle de polling par timeframe (secondes)
POLL_INTERVALS = {
    "1m": 30, "3m": 60, "5m": 90, "15m": 120,
    "30m": 180, "1h": 120, "4h": 300, "1d": 600,
}


class RealTimeDataCollector:
    """
    Collecte les données de marché en temps réel via polling REST.
    Poll Binance Futures public API toutes les N secondes.
    Déclenche les callbacks à chaque nouvelle bougie fermée.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db = MarketDatabase(config_path)

        # Buffers en mémoire (dernières N bougies par paire/TF)
        self.ohlcv_buffer: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=500))
        )
        self.last_candle_ts: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Callbacks pour notifier les stratégies
        self._callbacks: List[Callable] = []

        self.pairs = self.config["trading"]["pairs"]
        tf_config = self.config["trading"]["timeframes"]
        self.primary_tf = tf_config["primary"]
        # Collecter primary + higher + lower (pour micro-structure 15m)
        self.timeframes = [tf_config["primary"], tf_config["higher"]]
        if tf_config.get("lower"):
            self.timeframes.append(tf_config["lower"])
        self.running = False

        # CCXT REST pour polling (public, pas de clés nécessaires)
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        logger.info("Collecteur REST polling initialisé (Binance Futures public)")

    def register_callback(self, callback: Callable):
        """Enregistre un callback appelé à chaque nouvelle bougie fermée."""
        self._callbacks.append(callback)

    def get_latest_ohlcv(self, symbol: str, timeframe: str, n: int = 300) -> pd.DataFrame:
        """
        Retourne les N dernieres bougies.
        Combine DB historique + buffer live pour garantir assez de donnees.
        """
        # 1. Charger les donnees historiques depuis la DB
        df_hist = pd.DataFrame()
        try:
            from data.collectors.historical import HistoricalDataCollector
            collector = HistoricalDataCollector()
            df_hist = collector.load_data(symbol, timeframe)
        except Exception as e:
            logger.warning(f"Fallback DB echoue {symbol}/{timeframe}: {e}")

        # 2. Convertir le buffer live en DataFrame
        buffer = self.ohlcv_buffer[symbol][timeframe]
        df_live = pd.DataFrame()
        if buffer:
            candles = list(buffer)
            df_live = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], unit='ms', utc=True)
            df_live.set_index('timestamp', inplace=True)
            df_live = df_live.astype(float)

        # 3. Fusionner : historique + live (le live ecrase si meme timestamp)
        if not df_hist.empty and not df_live.empty:
            df = pd.concat([df_hist, df_live])
        elif not df_hist.empty:
            df = df_hist
        elif not df_live.empty:
            df = df_live
        else:
            return pd.DataFrame()

        # 4. Dedupliquer et trier, garder les N dernieres
        df = df[~df.index.duplicated(keep='last')].sort_index()
        return df.tail(n)

    def get_orderbook(self, symbol: str) -> dict:
        return {}

    async def _poll_ohlcv(self, symbol: str, timeframe: str):
        """Poll les bougies OHLCV toutes les N secondes."""
        interval = POLL_INTERVALS.get(timeframe, 60)
        logger.info(f"Polling OHLCV {symbol}/{timeframe} toutes les {interval}s")

        while self.running:
            try:
                # Récupère les 50 dernières bougies
                candles = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=50)
                )
                if not candles:
                    await asyncio.sleep(interval)
                    continue

                # Toutes les bougies sauf la dernière (en cours)
                closed = candles[:-1]
                # N'ajouter que les bougies plus recentes que le buffer actuel
                prev_ts = self.last_candle_ts[symbol].get(timeframe, 0)
                new_candles = [c for c in closed if c[0] > prev_ts]
                for c in new_candles:
                    self.ohlcv_buffer[symbol][timeframe].append(c)

                # Vérifier si une nouvelle bougie est apparue
                latest_ts = closed[-1][0] if closed else 0
                prev_ts = self.last_candle_ts[symbol].get(timeframe, 0)

                if latest_ts > prev_ts:
                    self.last_candle_ts[symbol][timeframe] = latest_ts
                    logger.info(
                        f"Nouvelle bougie {symbol}/{timeframe} : "
                        f"close={closed[-1][4]:.2f}"
                    )

                    # Sauvegarder en DB
                    df = pd.DataFrame(
                        [closed[-1]],
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    df.set_index('timestamp', inplace=True)
                    self.db.save_ohlcv(df, symbol, timeframe)

                    # Notifier uniquement sur le timeframe primary
                    if timeframe == self.primary_tf:
                        for cb in self._callbacks:
                            try:
                                cb(symbol, timeframe)
                            except Exception as e:
                                logger.error(f"Erreur callback : {e}")

            except Exception as e:
                logger.error(f"Erreur poll OHLCV {symbol}/{timeframe} : {e}")

            await asyncio.sleep(interval)

    async def start(self):
        """Lance le polling pour toutes les paires/TF."""
        self.running = True
        logger.info(f"Démarrage collecteur polling : {self.pairs}")

        # Initialiser les buffers avec les données historiques
        for symbol in self.pairs:
            for tf in self.timeframes:
                try:
                    candles = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda s=symbol, t=tf: self.exchange.fetch_ohlcv(s, t, limit=300)
                    )
                    for c in candles[:-1]:
                        self.ohlcv_buffer[symbol][tf].append(c)
                    if candles:
                        self.last_candle_ts[symbol][tf] = candles[-2][0] if len(candles) > 1 else 0
                    logger.info(f"Buffer initialisé {symbol}/{tf} : {len(candles)} bougies")
                except Exception as e:
                    logger.error(f"Erreur init buffer {symbol}/{tf} : {e}")

        # Lancer les polling en parallèle
        tasks = []
        for symbol in self.pairs:
            for tf in self.timeframes:
                tasks.append(self._poll_ohlcv(symbol, tf))

        await asyncio.gather(*tasks)

    async def stop(self):
        """Arrête le collecteur."""
        self.running = False
        logger.info("Collecteur polling arrêté.")
