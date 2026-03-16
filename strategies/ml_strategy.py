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
from utils.discord_notifier import DiscordNotifier

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
        self.tf_lower = self.config["trading"]["timeframes"].get("lower")
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

        # Discord notifier (optionnel — activé si webhook_url configuré)
        discord_cfg = self.config.get("discord", {})
        webhook_url = discord_cfg.get("webhook_url", "")
        self.discord: Optional[DiscordNotifier] = None
        if webhook_url and webhook_url.startswith("https://discord"):
            self.discord = DiscordNotifier(webhook_url)
            logger.info("Discord notifications activées ✓")
        else:
            logger.info("Discord notifications désactivées (pas de webhook dans config.yaml)")

        # Status périodique Discord (toutes les N minutes)
        self._status_interval = discord_cfg.get("status_interval_minutes", 60) * 60
        self._last_status_ts  = 0.0

    async def initialize(self):
        """Initialisation : données historiques + levier + warmup."""
        logger.info("=== Initialisation de la stratégie ===")

        # Configurer le levier pour chaque paire
        if self.config["trading"].get("market_type") == "futures":
            for pair in self.pairs:
                self.broker.set_leverage(pair, self.leverage)

        # Télécharger les données historiques de warmup
        logger.info("Téléchargement des données de warmup...")
        fetch_tfs = [self.tf_primary, self.tf_higher]
        if self.tf_lower:
            fetch_tfs.append(self.tf_lower)
        self.historical_collector.fetch_all_pairs(
            pairs=self.pairs,
            timeframes=fetch_tfs,
            days=30,  # 30 jours de warmup
        )

        # Pré-calculer les features pour initialiser les modèles
        for pair in self.pairs:
            await self._warmup_pair(pair)

        logger.info("=== Stratégie prête ! ===")

        # Notif démarrage Discord
        if self.discord:
            testnet = self.config.get("exchange", {}).get("sandbox", True)
            self.discord.notify_start(
                pairs=self.pairs,
                leverage=self.leverage,
                testnet=testnet,
            )

    async def _warmup_pair(self, symbol: str):
        """Charge et précalcule les features pour une paire."""
        df = self.historical_collector.load_data(symbol, self.tf_primary)
        df_higher = self.historical_collector.load_data(symbol, self.tf_higher)
        df_lower = pd.DataFrame()
        if self.tf_lower:
            df_lower = self.historical_collector.load_data(symbol, self.tf_lower)

        if len(df) >= self.warmup_candles:
            features = self.feature_engineer.compute_all(
                df,
                higher_tf_df=df_higher if not df_higher.empty else None,
                lower_tf_df=df_lower if not df_lower.empty else None,
            )
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

            # Récupérer le lower timeframe (15m) pour micro-structure
            df_lower = pd.DataFrame()
            if self.tf_lower:
                df_lower = self.realtime_collector.get_latest_ohlcv(
                    symbol, self.tf_lower, n=200
                )

            # Calculer les features (1h + 4h + 15m)
            features = self.feature_engineer.compute_all(
                df,
                higher_tf_df=df_higher if not df_higher.empty else None,
                lower_tf_df=df_lower if not df_lower.empty else None,
            )
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
            raw_scores = signal.get('raw_scores')   # dict SAC/PPO/DDPG si dispo

            current_price = float(df['close'].iloc[-1])

            # ── Notif bougie Discord ──────────────────────────────────────────
            if self.discord:
                self.discord.notify_candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    close=current_price,
                    direction=direction,
                    confidence=confidence,
                    regime=regime,
                    raw_scores=raw_scores,
                )
                # Status périodique
                import time as _time
                now = _time.time()
                if now - self._last_status_ts >= self._status_interval:
                    self._last_status_ts = now
                    self._send_status_discord()
            # ─────────────────────────────────────────────────────────────────

            # Verifier les SL/TP des positions ouvertes (trailing + partial)
            sl_tp_hit = self.risk_manager.check_sl_tp(symbol, current_price)
            if sl_tp_hit:
                self._execute_close(symbol, current_price, reason=sl_tp_hit)
                # Si la position a ete entierement fermee, reset le signal
                if symbol not in self.risk_manager.open_positions:
                    self.last_signal[symbol] = 0

            # Gerer la position en fonction du signal
            current_position = self.last_signal.get(symbol, 0)

            if direction != 0 and direction != current_position:
                # Nouveau signal différent → fermer si position ouverte puis entrer
                if current_position != 0:
                    self._execute_close(symbol, current_price, reason="signal_reversal")

                if position_size > 0:
                    multi_horizon = signal.get('multi_horizon')
                    self._execute_entry(
                        symbol, direction, confidence, position_size,
                        current_price, df,
                        raw_scores=raw_scores, regime=regime,
                        multi_horizon=multi_horizon,
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
        raw_scores: Optional[dict] = None,
        regime: Optional[str] = None,
        multi_horizon: Optional[dict] = None,
    ):
        """Execute une entree en position."""
        side = 'buy' if direction == 1 else 'sell'

        # Calculer la taille via le gestionnaire de risque (adapte au regime + multi-horizon Mamba)
        sizing = self.risk_manager.compute_position_size(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            ohlcv_df=ohlcv_df,
            regime=regime,
            multi_horizon=multi_horizon,
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
                'atr_value': sizing.get('atr_value', 0.0),
                'confidence': confidence,
                'regime': regime,
                'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
                'datetime': str(datetime.now(timezone.utc)),
            }
            self.risk_manager.register_trade(trade)
            self.db.save_trade({**trade, 'status': 'open'})

            # ── Notif Discord trade entry ─────────────────────────────────
            if self.discord:
                reason_parts = []
                if regime:
                    reason_parts.append(f"Régime: {regime}")
                if raw_scores:
                    top = sorted(raw_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
                    reason_parts.append("  ".join([f"{k}={v:.2f}" for k, v in top]))
                self.discord.notify_trade_entry(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    quantity=quantity,
                    usdt_amount=sizing['usdt_amount'],
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reason="  •  ".join(reason_parts) if reason_parts else "",
                )
            # ─────────────────────────────────────────────────────────────

    def _execute_close(self, symbol: str, current_price: float, reason: str = "signal"):
        """
        Execute la fermeture d'une position (totale ou partielle).
        Pour les partial TP, on ne ferme qu'une fraction de la quantite.
        """
        is_partial = reason.startswith("partial_tp_")

        if is_partial:
            # Partial close : on ferme juste la quantite indiquee par le risk manager
            pos = self.risk_manager.open_positions.get(symbol)
            if not pos:
                return
            qty_to_close = pos.get('partial_close_qty', 0)
            if qty_to_close <= 0:
                return

            # Placer un market order pour la quantite partielle
            side_close = 'sell' if pos['side'] == 'buy' else 'buy'
            order = self.broker.place_market_order(symbol, side_close, qty_to_close)
        else:
            # Full close : annuler tous les ordres et fermer toute la position
            self.broker.cancel_all_orders(symbol)
            order = self.broker.close_position(symbol)

        if order:
            result = self.risk_manager.close_trade(symbol, current_price, reason)
            if result:
                status = 'partial_close' if result.get('is_partial') else 'closed'
                self.db.save_trade({**result, 'status': status})

                # Notif Discord trade close
                if self.discord:
                    pnl_usdt = result.get('pnl_usdt', 0.0)
                    pnl_pct  = result.get('pnl_pct', 0.0)
                    self.discord.notify_trade_close(
                        symbol=symbol,
                        side=result.get('side', '?'),
                        entry_price=result.get('entry_price', 0.0),
                        close_price=current_price,
                        pnl_usdt=pnl_usdt,
                        pnl_pct=pnl_pct,
                        reason=reason,
                    )

    def _send_status_discord(self):
        """Envoie le status périodique sur Discord."""
        try:
            balance = self.broker.get_balance()
            # broker.get_balance() retourne {'free': X, 'total': X} directement
            balance_usdt = float(balance.get('free', balance.get('USDT', {}).get('free', 0.0))) if isinstance(balance, dict) else 0.0
            status = self.risk_manager.get_status()
            open_positions = {}
            for sym in self.pairs:
                pos = self.risk_manager.open_positions.get(sym)
                if pos:
                    open_positions[sym] = pos
            self.discord.notify_status(
                balance_usdt=balance_usdt,
                open_positions=open_positions,
                total_trades=status.get('total_trades', 0),
                win_rate=status.get('win_rate', 0.0),
                total_pnl=status.get('total_pnl', 0.0),
            )
        except Exception as e:
            logger.warning(f"Erreur status Discord : {e}")

    async def _monitor_sl_tp(self):
        """
        Monitore le SL/TP toutes les 30s (client-side).
        Remplace les ordres serveur STOP_MARKET/TAKE_PROFIT_MARKET
        qui ne sont pas supportés sur le testnet Binance.
        """
        import ccxt
        exchange = ccxt.binance({"options": {"defaultType": "future"}})

        while self.realtime_collector.running:
            try:
                for symbol in self.pairs:
                    if symbol not in self.risk_manager.open_positions:
                        continue
                    # Récupérer le prix actuel
                    ticker = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=symbol: exchange.fetch_ticker(s)
                    )
                    price = float(ticker['last'])

                    # Vérifier SL/TP
                    hit = self.risk_manager.check_sl_tp(symbol, price)
                    if hit:
                        logger.info(f"[MONITOR] {symbol} {hit} declenche @ {price:.2f}")
                        self._execute_close(symbol, price, reason=hit)

            except Exception as e:
                logger.debug(f"Erreur monitor SL/TP : {e}")

            await asyncio.sleep(30)

    async def run(self):
        """Loop principal de trading."""
        logger.info("=== Démarrage du bot de trading ===")
        await self.initialize()

        # Enregistrer le callback de nouvelles bougies
        self.realtime_collector.register_callback(self.on_new_candle)

        # Démarrer le collecteur temps réel + monitor SL/TP en parallèle
        logger.info("Démarrage du collecteur WebSocket...")
        await asyncio.gather(
            self.realtime_collector.start(),
            self._monitor_sl_tp(),
        )

    def stop(self):
        """Arrêt propre du bot."""
        logger.info("Arrêt du bot...")
        asyncio.create_task(self.realtime_collector.stop())
        status = self.risk_manager.get_status()
        logger.info(f"Bilan final : {status}")
        if self.discord:
            self.discord.notify_stop(
                total_pnl=status.get('total_pnl', 0.0),
                total_trades=status.get('total_trades', 0),
            )
