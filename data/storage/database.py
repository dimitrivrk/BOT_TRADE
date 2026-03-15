"""
Couche de persistance - SQLite (extensible vers PostgreSQL/TimescaleDB).
Stocke les OHLCV, les features calculées et les trades.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
import yaml

from utils.logger import setup_logger

logger = setup_logger("data.storage.database")


class MarketDatabase:
    """
    Interface de base de données pour les données de marché.
    Supporte SQLite nativement, extensible vers TimescaleDB.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.db_path = config["data"]["db_path"]
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"Base de données initialisée : {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")      # écriture concurrente
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        return conn

    def _init_db(self):
        """Crée les tables si elles n'existent pas."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    timeframe   TEXT NOT NULL,
                    timestamp   INTEGER NOT NULL,
                    datetime    TEXT NOT NULL,
                    open        REAL NOT NULL,
                    high        REAL NOT NULL,
                    low         REAL NOT NULL,
                    close       REAL NOT NULL,
                    volume      REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
                    ON ohlcv(symbol, timeframe, timestamp);

                CREATE TABLE IF NOT EXISTS features (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    timeframe   TEXT NOT NULL,
                    timestamp   INTEGER NOT NULL,
                    feature_json TEXT NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       INTEGER NOT NULL,
                    datetime        TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    quantity        REAL NOT NULL,
                    price           REAL NOT NULL,
                    fee             REAL DEFAULT 0.0,
                    pnl             REAL DEFAULT 0.0,
                    strategy        TEXT DEFAULT 'ensemble',
                    confidence      REAL DEFAULT 0.0,
                    stop_loss       REAL,
                    take_profit     REAL,
                    status          TEXT DEFAULT 'open'
                );

                CREATE TABLE IF NOT EXISTS model_predictions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   INTEGER NOT NULL,
                    symbol      TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    direction   TEXT NOT NULL,
                    confidence  REAL NOT NULL,
                    horizon     INTEGER NOT NULL,
                    actual_return REAL
                );
            """)

    # -------------------------------------------------------------------------
    # OHLCV
    # -------------------------------------------------------------------------

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Sauvegarde un DataFrame OHLCV (upsert)."""
        if df.empty:
            return

        records = []
        for ts, row in df.iterrows():
            ts_ms = int(ts.timestamp() * 1000) if hasattr(ts, 'timestamp') else int(ts)
            records.append((
                symbol, timeframe, ts_ms,
                str(pd.Timestamp(ts_ms, unit='ms')),
                float(row['open']), float(row['high']),
                float(row['low']), float(row['close']),
                float(row['volume']),
            ))

        with self._get_connection() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO ohlcv
                   (symbol, timeframe, timestamp, datetime, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                records,
            )
        logger.debug(f"Sauvegardé {len(records)} bougies {symbol}/{timeframe}")

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Charge les OHLCV depuis la base."""
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND timeframe=?"
        params = [symbol, timeframe]

        if start:
            start_ms = int(pd.Timestamp(start).timestamp() * 1000)
            query += " AND timestamp >= ?"
            params.append(start_ms)
        if end:
            end_ms = int(pd.Timestamp(end).timestamp() * 1000)
            query += " AND timestamp <= ?"
            params.append(end_ms)

        query += " ORDER BY timestamp"
        if limit:
            query += f" LIMIT {limit}"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df

    def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """Retourne le timestamp de la dernière bougie en base (en ms)."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT MAX(timestamp) FROM ohlcv WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
        return result[0] if result and result[0] else None

    # -------------------------------------------------------------------------
    # TRADES
    # -------------------------------------------------------------------------

    def save_trade(self, trade: dict):
        """Enregistre un trade exécuté."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO trades
                   (timestamp, datetime, symbol, side, quantity, price, fee, pnl,
                    strategy, confidence, stop_loss, take_profit, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.get('timestamp', int(datetime.now().timestamp() * 1000)),
                    trade.get('datetime', str(datetime.now())),
                    trade['symbol'], trade['side'],
                    trade['quantity'], trade['price'],
                    trade.get('fee', 0.0), trade.get('pnl', 0.0),
                    trade.get('strategy', 'ensemble'),
                    trade.get('confidence', 0.0),
                    trade.get('stop_loss'), trade.get('take_profit'),
                    trade.get('status', 'open'),
                ),
            )

    def load_trades(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Charge l'historique des trades."""
        query = "SELECT * FROM trades"
        params = []
        if symbol:
            query += " WHERE symbol=?"
            params.append(symbol)
        query += " ORDER BY timestamp"

        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # -------------------------------------------------------------------------
    # PRÉDICTIONS
    # -------------------------------------------------------------------------

    def save_prediction(self, prediction: dict):
        """Sauvegarde une prédiction de modèle pour monitoring."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO model_predictions
                   (timestamp, symbol, model, direction, confidence, horizon)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    prediction['timestamp'], prediction['symbol'],
                    prediction['model'], prediction['direction'],
                    prediction['confidence'], prediction['horizon'],
                ),
            )
