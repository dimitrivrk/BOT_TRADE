"""
Collecteur de données en temps réel via WebSocket CCXT.
Flux live : OHLCV, orderbook, trades, funding rate, OI.
"""

import asyncio
import ccxt.pro as ccxtpro
import pandas as pd
from typing import Dict, List, Callable, Optional
from collections import defaultdict, deque
import yaml
import os
from dotenv import load_dotenv

from data.storage.database import MarketDatabase
from utils.logger import setup_logger

load_dotenv("config/.env")
logger = setup_logger("data.collectors.market_data")


class RealTimeDataCollector:
    """
    Collecte les données de marché en temps réel via WebSocket.
    Maintient un buffer en mémoire (rolling window) des dernières bougies.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db = MarketDatabase(config_path)
        self.exchange: Optional[ccxtpro.Exchange] = None

        # Buffers en mémoire (dernières N bougies par paire/TF)
        self.ohlcv_buffer: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=500))
        )
        self.orderbook_buffer: Dict[str, dict] = {}
        self.ticker_buffer: Dict[str, dict] = {}

        # Callbacks pour notifier les stratégies
        self._callbacks: List[Callable] = []

        self.pairs = self.config["trading"]["pairs"]
        tf_config = self.config["trading"]["timeframes"]
        self.timeframes = [tf_config["primary"], tf_config["higher"], tf_config["lower"]]
        self.running = False

    def _init_exchange(self) -> ccxtpro.Exchange:
        exc_config = self.config["exchange"]
        name = exc_config["name"]

        exchange_class = getattr(ccxtpro, name)

        # Utiliser testnet ou mainnet selon config
        if exc_config.get("sandbox", True):
            api_key = os.getenv(f"{name.upper()}_TESTNET_API_KEY", "")
            secret = os.getenv(f"{name.upper()}_TESTNET_SECRET_KEY", "")
        else:
            api_key = os.getenv(f"{name.upper()}_API_KEY", "")
            secret = os.getenv(f"{name.upper()}_SECRET_KEY", "")

        exchange = exchange_class({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": self.config["trading"].get("market_type", "spot")},
        })

        if exc_config.get("sandbox") and hasattr(exchange, 'set_sandbox_mode'):
            exchange.set_sandbox_mode(True)

        return exchange

    def register_callback(self, callback: Callable):
        """Enregistre un callback appelé à chaque nouvelle bougie fermée."""
        self._callbacks.append(callback)

    def get_latest_ohlcv(self, symbol: str, timeframe: str, n: int = 200) -> pd.DataFrame:
        """Retourne les N dernières bougies depuis le buffer mémoire."""
        buffer = self.ohlcv_buffer[symbol][timeframe]
        if not buffer:
            return pd.DataFrame()

        candles = list(buffer)[-n:]
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.astype(float)

    def get_orderbook(self, symbol: str) -> dict:
        """Retourne le carnet d'ordres le plus récent."""
        return self.orderbook_buffer.get(symbol, {})

    async def _watch_ohlcv(self, symbol: str, timeframe: str):
        """Écoute les bougies OHLCV en temps réel pour une paire/TF."""
        logger.info(f"Démarrage watch OHLCV : {symbol}/{timeframe}")
        while self.running:
            try:
                candles = await self.exchange.watch_ohlcv(symbol, timeframe)
                for candle in candles:
                    self.ohlcv_buffer[symbol][timeframe].append(candle)

                # Sauvegarder la bougie fermée en DB (la pénultième = fermée)
                buf = self.ohlcv_buffer[symbol][timeframe]
                if len(buf) >= 2:
                    closed_candle = list(buf)[-2]
                    df = pd.DataFrame(
                        [closed_candle],
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    df.set_index('timestamp', inplace=True)
                    self.db.save_ohlcv(df, symbol, timeframe)

                # Notifier les stratégies
                for cb in self._callbacks:
                    try:
                        await asyncio.coroutine(cb)(symbol, timeframe) if asyncio.iscoroutinefunction(cb) else cb(symbol, timeframe)
                    except Exception as e:
                        logger.error(f"Erreur callback : {e}")

            except Exception as e:
                logger.error(f"Erreur watch OHLCV {symbol}/{timeframe} : {e}")
                await asyncio.sleep(5)

    async def _watch_orderbook(self, symbol: str, depth: int = 10):
        """Écoute le carnet d'ordres."""
        while self.running:
            try:
                ob = await self.exchange.watch_order_book(symbol, depth)
                self.orderbook_buffer[symbol] = {
                    "bids": ob["bids"][:depth],
                    "asks": ob["asks"][:depth],
                    "timestamp": ob.get("timestamp"),
                }
            except Exception as e:
                logger.error(f"Erreur watch orderbook {symbol} : {e}")
                await asyncio.sleep(5)

    async def _watch_ticker(self, symbol: str):
        """Écoute le ticker en temps réel."""
        while self.running:
            try:
                ticker = await self.exchange.watch_ticker(symbol)
                self.orderbook_buffer[symbol] = ticker
            except Exception as e:
                logger.error(f"Erreur watch ticker {symbol} : {e}")
                await asyncio.sleep(5)

    async def start(self):
        """Lance tous les flux WebSocket en parallèle."""
        self.exchange = self._init_exchange()
        self.running = True
        logger.info(f"Démarrage collecteur temps réel : {self.pairs}")

        tasks = []
        ob_depth = self.config["features"]["advanced"].get("orderbook_depth", 10)

        for symbol in self.pairs:
            for tf in self.timeframes:
                tasks.append(self._watch_ohlcv(symbol, tf))
            tasks.append(self._watch_orderbook(symbol, ob_depth))

        await asyncio.gather(*tasks)

    async def stop(self):
        """Arrête proprement les connexions WebSocket."""
        self.running = False
        if self.exchange:
            await self.exchange.close()
        logger.info("Collecteur temps réel arrêté.")
