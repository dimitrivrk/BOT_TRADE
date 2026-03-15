"""
Stratégie ML principale : orchestre les modèles IA + l'exécution.
Loop de trading asynchrone principal.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timezone
import yaml

from data.collectors.market_data import RealTimeDataCollector
from data.collectors.historical import HistoricalDataCollector
from data.processors.features import FeatureEngineer
from data.processors.indicators import market_regime
from models.ensemble import EnsemblePredictor
from execution.broker import ExchangeBroker
from execution.risk_manager import RiskManager
from data.storage.database import MarketDatabase
from utils.logger import setup_logger

logger = setup_logger("strategies.ml_strategy")


class MLTradingStrategy:
    """
    Stratégie de trading IA principale.
    Orchestre : collecte → features → signaux → risque → exécution → monitoring.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.pairs = self.config["trading"]["pairs"]
        self.tf_primary = self.config["trading"]["timeframes"]["primary"]
        self.tf_higher = self.config["trading"]["timeframes"]["higher"]
        self.leverage = self.config["trading"].get("leverage", 1)

        # Composants
        self.realtime_collector = RealTimeDataCollector(config_path)
        self.historical_collector = HistoricalDataCollector(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.ensemble = EnsemblePredictor(config_path)
        self.broker = ExchangeBroker(config_path)
        self.risk_manager = RiskManager(config_path)
        self.db = MarketDatabase(config_path)

        # Warmup : N bougies nécessaires avant de trader
        self.warmup_candles = self.config["data"]["warmup_candles"]
        self.is_ready: Dict[str, bool] = {pair: False for pair in self.pairs}

        # État des signaux précédents (éviter les signaux répétés)
        self.last_signal: Dict[str, int] = {pair: 0 for pair in self.pairs}

    async def initialize(self):
        """Initialisation : données historiques + levier + warmup."""
        logger.info("=== Initialisation de la stratégie ===")

        # Configurer le levier pour chaque paire
        if self.config["trading"].get("market_type") == "futures":
            for pair in self.pairs:
                self.broker.set_leverage(pair, self.leverage)

        # Télécharger les données historiques de warmup
        logger.info("Téléchargement des données de warmup...")
        self.historical_collector.fetch_all_pairs(
            pairs=self.pairs,
            timeframes=[self.tf_primary, self.tf_higher],
            days=30,  # 30 jours de warmup
        )

        # Pré-calculer les features pour initialiser les modèles
        for pair in self.pairs:
            await self._warmup_pair(pair)

        logger.info("=== Stratégie prête ! ===")

    async def _warmup_pair(self, symbol: str):
        """Charge et précalcule les features pour une paire."""
        df = self.historical_collector.load_data(symbol, self.tf_primary)
        df_higher = self.historical_collector.load_data(symbol, self.tf_higher)

        if len(df) >= self.warmup_candles:
            features = self.feature_engineer.compute_all(df, higher_tf_df=df_higher)
            if len(features) >= 50:
                self.is_ready[symbol] = True
                logger.info(f"✓ {symbol} prêt ({len(features)} features calculées)")
            else:
                logger.warning(f"⚠ {symbol} : features insuffisantes")
        else:
            logger.warning(f"⚠ {symbol} : données insuffisantes ({len(df)}/{self.warmup_candles} bougies)")

    def on_new_candle(self, symbol: str, timeframe: str):
        """
        Callback appelé à chaque nouvelle bougie fermée.
        Point d'entrée principal de la stratégie.
        """
        if timeframe != self.tf_primary:
            return  # on trade uniquement sur le TF principal

        if not self.is_ready.get(symbol, False):
            return

        if self.risk_manager.bot_stopped:
            logger.warning("Bot arrêté, trading suspendu.")
            return

        try:
            # Récupérer les données actuelles
            df = self.realtime_collector.get_latest_ohlcv(
                symbol, timeframe, n=300
            )
            if df.empty or len(df) < 100:
                return

            df_higher = self.realtime_collector.get_latest_ohlcv(
                symbol, self.tf_higher, n=100
            )

            # Calculer les features
            features = self.feature_engineer.compute_all(df, higher_tf_df=df_higher if not df_higher.empty else None)
            if features.empty:
                return

            # Détecter le régime de marché
            regime = None
            if 'regime_trending_bull' in features.columns:
                last = features.iloc[-1]
                if last.get('regime_trending_bull', 0) > 0.5:
                    regime = 'trending_bull'
                elif last.get('regime_trending_bear', 0) > 0.5:
                    regime = 'trending_bear'
                elif last.get('regime_ranging', 0) > 0.5:
                    regime = 'ranging'
                elif last.get('regime_volatile', 0) > 0.5:
                    regime = 'volatile'

            # Obtenir le signal de l'ensemble
            signal = self.ensemble.predict(features, df, symbol, regime=regime)

            if not signal:
                return

            direction = signal['direction']
            confidence = signal['confidence']
            position_size = signal['position_size']

            current_price = float(df['close'].iloc[-1])

            # Vérifier les SL/TP des positions ouvertes
            sl_tp_hit = self.risk_manager.check_sl_tp(symbol, current_price)
            if sl_tp_hit:
                self._execute_close(symbol, current_price, reason=sl_tp_hit)

            # Gérer la position en fonction du signal
            current_position = self.last_signal.get(symbol, 0)

            if direction != 0 and direction != current_position:
                # Nouveau signal différent → fermer si position ouverte puis entrer
                if current_position != 0:
                    self._execute_close(symbol, current_price, reason="signal_reversal")

                if position_size > 0:
                    self._execute_entry(
                        symbol, direction, confidence, position_size,
                        current_price, df
                    )
                    self.last_signal[symbol] = direction

            elif direction == 0 and current_position != 0:
                # Signal neutre → sortir
                self._execute_close(symbol, current_price, reason="neutral_signal")
                self.last_signal[symbol] = 0

            # Sauvegarder la prédiction
            self.db.save_prediction({
                'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
                'symbol': symbol,
                'model': 'ensemble',
                'direction': str(direction),
                'confidence': confidence,
                'horizon': 1,
            })

        except Exception as e:
            logger.error(f"Erreur on_new_candle {symbol} : {e}", exc_info=True)

    def _execute_entry(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        position_size_pct: float,
        entry_price: float,
        ohlcv_df: pd.DataFrame,
    ):
        """Exécute une entrée en position."""
        side = 'buy' if direction == 1 else 'sell'

        # Calculer la taille via le gestionnaire de risque
        sizing = self.risk_manager.compute_position_size(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            ohlcv_df=ohlcv_df,
        )

        if not sizing:
            logger.warning(f"Position sizing refusé pour {symbol}")
            return

        quantity = sizing['quantity']
        stop_loss = sizing['stop_loss']
        take_profit = sizing['take_profit']

        # Annuler les ordres précédents
        self.broker.cancel_all_orders(symbol)

        # Placer l'ordre d'entrée (market pour réactivité)
        order = self.broker.place_market_order(symbol, side, quantity)

        if order:
            # Placer SL et TP
            self.broker.place_stop_loss(symbol, side, quantity, stop_loss)
            self.broker.place_take_profit(symbol, side, quantity, take_profit)

            # Enregistrer dans le risk manager
            trade = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'usdt_amount': sizing['usdt_amount'],
                'confidence': confidence,
                'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
                'datetime': str(datetime.now(timezone.utc)),
            }
            self.risk_manager.register_trade(trade)
            self.db.save_trade({**trade, 'status': 'open'})

    def _execute_close(self, symbol: str, current_price: float, reason: str = "signal"):
        """Exécute la fermeture d'une position."""
        self.broker.cancel_all_orders(symbol)
        order = self.broker.close_position(symbol)

        if order:
            result = self.risk_manager.close_trade(symbol, current_price, reason)
            if result:
                self.db.save_trade({**result, 'status': 'closed'})

    async def run(self):
        """Loop principal de trading."""
        logger.info("=== Démarrage du bot de trading ===")
        await self.initialize()

        # Enregistrer le callback de nouvelles bougies
        self.realtime_collector.register_callback(self.on_new_candle)

        # Démarrer le collecteur temps réel
        logger.info("Démarrage du collecteur WebSocket...")
        await self.realtime_collector.start()

    def stop(self):
        """Arrêt propre du bot."""
        logger.info("Arrêt du bot...")
        asyncio.create_task(self.realtime_collector.stop())
        logger.info(f"Bilan final : {self.risk_manager.get_status()}")
