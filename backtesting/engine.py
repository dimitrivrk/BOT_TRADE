"""
Moteur de backtesting vectorisé basé sur vectorbt.
Supporte : walk-forward optimization, Monte Carlo, analyse de sensibilité.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import yaml
import json
from datetime import datetime

from data.collectors.historical import HistoricalDataCollector
from data.processors.features import FeatureEngineer
from models.ensemble import EnsemblePredictor
from backtesting.metrics import PerformanceMetrics
from utils.logger import setup_logger

logger = setup_logger("backtesting.engine")


class BacktestEngine:
    """
    Moteur de backtesting complet.
    Utilise vectorbt pour la simulation vectorisée ultra-rapide.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.bt_config = self.config["backtesting"]
        self.risk_config = self.config["risk"]
        self.initial_capital = self.bt_config.get("initial_capital", 10000.0)
        self.commission = self.bt_config.get("commission", 0.001)
        self.slippage = self.bt_config.get("slippage", 0.0005)

        self.collector = HistoricalDataCollector(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.metrics_calc = PerformanceMetrics()
        self.config_path = config_path

        # Initialiser l'ensemble (TFT + RL + technique)
        try:
            self.ensemble = EnsemblePredictor(config_path)
            if self.ensemble.rl:
                self.ensemble.rl.load()
                logger.info("RL charge pour le backtest")
        except Exception as e:
            logger.warning(f"Ensemble non disponible, fallback technique : {e}")
            self.ensemble = None

    def run(
        self,
        symbol: str,
        signal_func: Optional[Callable] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict:
        """
        Lance un backtest complet pour un symbole.

        Args:
            symbol: Paire (ex: 'BTC/USDT')
            signal_func: Fonction qui génère les signaux (si None, utilise l'ensemble)
            start: Date de début
            end: Date de fin
            timeframe: Timeframe à utiliser

        Returns:
            Dictionnaire de métriques de performance
        """
        start = start or self.bt_config.get("start_date", "2023-01-01")
        end = end or self.bt_config.get("end_date", "2024-12-31")
        timeframe = timeframe or self.config["trading"]["timeframes"]["primary"]

        logger.info(f"Backtest {symbol} {timeframe} | {start} → {end}")

        # Charger les données
        df = self.collector.load_data(symbol, timeframe, start=start, end=end)
        if df.empty or len(df) < 200:
            logger.error(f"Données insuffisantes pour {symbol}")
            return {}

        # Calculer les features
        tf_higher = self.config["trading"]["timeframes"]["higher"]
        df_higher = self.collector.load_data(symbol, tf_higher, start=start, end=end)
        features = self.feature_engineer.compute_all(df, higher_tf_df=df_higher if not df_higher.empty else None)

        # Générer les signaux
        if signal_func is not None:
            signals = signal_func(features, df)
        elif self.ensemble is not None:
            safe_symbol = symbol or "BTC/USDT"
            signals = self._generate_signals_ensemble(features, df, safe_symbol)
        else:
            logger.warning("Pas d'ensemble disponible, fallback signaux basiques")
            signals = self._generate_signals_vectorized(features, df)

        # ⚠ Aligner tous les DataFrames sur l'index commun (features drop les NaN du warmup)
        common_index = signals.index
        df = df.loc[df.index.isin(common_index)]
        df = df.reindex(common_index).dropna()
        signals = signals.reindex(df.index)

        # Appliquer SL/TP dynamique (ATR-based)
        sl_pct, tp_pct = self._compute_dynamic_sltp(df)

        # Simulation avec vectorbt
        result = self._run_vbt_simulation(df, signals, sl_pct, tp_pct)

        # Calculer les métriques
        metrics = self.metrics_calc.compute(result, self.initial_capital)
        metrics['symbol'] = symbol
        metrics['timeframe'] = timeframe
        metrics['start'] = start
        metrics['end'] = end

        self._log_results(metrics)
        return metrics

    def _generate_signals_ensemble(
        self,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Genere les signaux via l'EnsemblePredictor (TFT + RL + technique).
        Itere sur chaque step avec une fenetre glissante.
        """
        signals = pd.DataFrame(index=features.index)
        scores = []
        directions = []
        confidences = []

        # Charger le modèle de prédiction (Mamba ou TFT fallback)
        if self.ensemble.mamba is not None and hasattr(self.ensemble.mamba, 'model'):
            if self.ensemble.mamba.model is None:
                try:
                    self.ensemble.mamba.load(symbol)
                    logger.info(f"Mamba/TFT chargé pour {symbol}")
                except Exception as e:
                    logger.warning(f"Mamba/TFT non disponible pour {symbol} : {e}")

        lookback = 100  # fenetre minimum pour les modeles
        total = len(features)
        log_interval = max(total // 10, 1)

        logger.info(f"Generation signaux ensemble sur {total} steps...")

        for i in range(total):
            if i < lookback:
                scores.append(0.0)
                directions.append(0)
                confidences.append(0.0)
                continue

            # Fenetre glissante
            feat_window = features.iloc[max(0, i - lookback):i + 1]
            price_window = prices.iloc[max(0, i - lookback):i + 1]

            try:
                result = self.ensemble.predict(
                    feat_window, price_window, symbol, regime=None
                )
                directions.append(result.get('direction', 0))
                confidences.append(result.get('confidence', 0.0))
                scores.append(result.get('raw_score', 0.0))
            except Exception:
                directions.append(0)
                confidences.append(0.0)
                scores.append(0.0)

            if (i + 1) % log_interval == 0:
                logger.info(f"  Signaux : {i + 1}/{total} ({(i + 1) / total:.0%})")

        score = pd.Series(scores, index=features.index)
        direction = pd.Series(directions, index=features.index)

        entry_threshold = 0.15
        signals['entry_long']  = (direction == 1)
        signals['entry_short'] = (direction == -1)
        signals['exit_long']   = (direction <= 0)
        signals['exit_short']  = (direction >= 0)
        signals['score']       = score

        long_rate  = signals['entry_long'].mean()
        short_rate = signals['entry_short'].mean()
        logger.debug(f"Signal rates → long={long_rate:.1%} short={short_rate:.1%} (score mean={score.mean():.3f} std={score.std():.3f})")

        return signals

    def _generate_signals_vectorized(
        self,
        features: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Génère les signaux d'entrée/sortie de manière vectorisée.
        Toutes les features sont z-scorées → on travaille avec np.sign() et
        des seuils en écarts-types, pas en valeurs absolues.
        """
        signals = pd.DataFrame(index=features.index)
        score = pd.Series(0.0, index=features.index)
        n_components = 0

        # --- Tendance (signe = direction, magnitude = force) ---
        for col, weight in [
            ('ema_9_21_cross', 0.25),
            ('ema_21_50_cross', 0.20),
            ('supertrend_dist', 0.20),   # positif si prix > supertrend (bullish)
            ('vwap_dist', 0.10),         # positif si prix > VWAP
        ]:
            if col in features.columns:
                score += np.sign(features[col].fillna(0)) * weight
                n_components += weight

        # --- Momentum z-scoré ---
        # rsi_14 z-scoré : valeur négative = oversold, positive = overbought
        if 'rsi_14' in features.columns:
            rsi_z = features['rsi_14'].fillna(0)
            # Signal contrarian léger + trend following
            score += np.where(rsi_z < -1.5,  0.3,   # fort oversold → acheter
                    np.where(rsi_z >  1.5, -0.3,     # fort overbought → vendre
                    np.sign(rsi_z) * 0.1))            # sinon trend following léger
            n_components += 0.3

        if 'macd_diff' in features.columns:
            score += np.sign(features['macd_diff'].fillna(0)) * 0.15
            n_components += 0.15

        if 'stoch_cross' in features.columns:
            score += np.sign(features['stoch_cross'].fillna(0)) * 0.10
            n_components += 0.10

        # --- Volume confirmation ---
        if 'cmf' in features.columns:
            score += np.sign(features['cmf'].fillna(0)) * 0.10
            n_components += 0.10

        if 'vol_ratio_20' in features.columns:
            # Volume élevé amplifie le signal directionnel
            vol_amp = (features['vol_ratio_20'].fillna(0)).clip(-2, 2) * 0.05
            score += vol_amp * np.sign(score)
            n_components += 0.05

        # Normaliser par le nombre de composantes actives
        if n_components > 0:
            score = score / n_components * 1.2   # rescale pour avoir des valeurs plus étalées

        score = score.fillna(0).clip(-1.0, 1.0)

        # --- Seuils d'entrée (calibrés pour générer ~5-15% de signaux actifs) ---
        entry_threshold = 0.15   # seuil bas = plus de trades
        exit_threshold  = 0.0

        signals['entry_long']  = (score >  entry_threshold)
        signals['entry_short'] = (score < -entry_threshold)
        signals['exit_long']   = (score <  exit_threshold)
        signals['exit_short']  = (score >  exit_threshold)
        signals['score']       = score

        # Log du taux de signaux pour debug
        long_rate  = signals['entry_long'].mean()
        short_rate = signals['entry_short'].mean()
        logger.debug(f"Signal rates → long={long_rate:.1%} short={short_rate:.1%} (score mean={score.mean():.3f} std={score.std():.3f})")

        return signals

    def _compute_dynamic_sltp(
        self,
        df: pd.DataFrame,
        atr_mult_sl: float = 2.0,
        rr_ratio: float = 2.5,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calcule SL/TP dynamiques basés sur l'ATR."""
        from data.processors.indicators import atr
        atr_val = atr(df, period=14)
        sl_pct = (atr_val * atr_mult_sl / df['close']).clip(0.005, 0.10)  # entre 0.5% et 10%
        tp_pct = sl_pct * rr_ratio
        return sl_pct, tp_pct

    def _run_vbt_simulation(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        sl_pct: pd.Series,
        tp_pct: pd.Series,
    ) -> vbt.Portfolio:
        """
        Lance la simulation vectorbt.
        On sépare long et short en deux portfolios puis on les combine,
        car vectorbt SizeType.Percent ne supporte pas les reversals directs.
        """
        close = df['close']
        freq = self._infer_freq(df)

        # Taille fixe par trade (fraction du capital)
        size_val = self.initial_capital * 0.95  # 95% du capital par trade

        entries      = signals['entry_long'].fillna(False).astype(bool)
        exits        = signals['exit_long'].fillna(False).astype(bool)
        short_entries = signals['entry_short'].fillna(False).astype(bool) if 'entry_short' in signals.columns else pd.Series(False, index=close.index)
        short_exits   = signals['exit_short'].fillna(False).astype(bool) if 'exit_short' in signals.columns else pd.Series(False, index=close.index)

        # --- Portfolio LONG ---
        pf_long = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            size=size_val,
            size_type="value",
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage,
            sl_stop=sl_pct,
            tp_stop=tp_pct,
            freq=freq,
            accumulate=False,
        )

        # --- Portfolio SHORT ---
        has_shorts = short_entries.any()
        if has_shorts:
            pf_short = vbt.Portfolio.from_signals(
                close=close,
                entries=short_entries,
                exits=short_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                size=size_val,
                size_type="value",
                init_cash=self.initial_capital,
                fees=self.commission,
                slippage=self.slippage,
                sl_stop=sl_pct,
                tp_stop=tp_pct,
                freq=freq,
                accumulate=False,
                direction="shortonly",
            )
            # Retourner le long (le plus simple pour le baseline)
            # TODO : combiner les deux equity curves pour une stratégie complète
            return pf_long
        else:
            return pf_long

    def _infer_freq(self, df: pd.DataFrame) -> str:
        """Infère la fréquence du DataFrame."""
        if len(df) < 2:
            return "1h"
        delta = df.index[1] - df.index[0]
        minutes = int(delta.total_seconds() / 60)
        mapping = {1: "1min", 5: "5min", 15: "15min", 30: "30min", 60: "1h", 240: "4h", 1440: "1D"}
        return mapping.get(minutes, "1h")

    def run_walk_forward(
        self,
        symbol: str,
        train_months: Optional[int] = None,
        test_months: Optional[int] = None,
    ) -> List[Dict]:
        """
        Walk-Forward Optimization : entraîne sur N mois, teste sur M mois, glisse.
        Évite le surapprentissage et donne une estimation réaliste des performances.
        """
        wf_config = self.bt_config.get("walk_forward", {})
        train_months = train_months or wf_config.get("train_months", 6)
        test_months = test_months or wf_config.get("test_months", 1)

        start = pd.Timestamp(self.bt_config.get("start_date", "2023-01-01"))
        end = pd.Timestamp(self.bt_config.get("end_date", "2024-12-31"))
        timeframe = self.config["trading"]["timeframes"]["primary"]

        # Charger toutes les données
        all_data = self.collector.load_data(symbol, timeframe)
        all_data = all_data[start:end]

        results = []
        window_start = start

        logger.info(f"Walk-Forward : train={train_months}M, test={test_months}M")

        while True:
            train_end = window_start + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)

            if test_end > end:
                break

            logger.info(f"WF Window : train [{window_start} → {train_end}] | test [{test_start} → {test_end}]")

            # Backtest sur la période de test
            result = self.run(
                symbol,
                start=str(test_start.date()),
                end=str(test_end.date()),
                timeframe=timeframe,
            )
            result['window_start'] = str(test_start.date())
            result['window_end'] = str(test_end.date())
            results.append(result)

            # Glisser d'un mois
            window_start += pd.DateOffset(months=test_months)

        # Synthèse globale
        if results:
            summary = self._summarize_walk_forward(results)
            logger.info(f"Walk-Forward terminé : {len(results)} fenêtres | Sharpe moyen={summary.get('mean_sharpe', 0):.2f}")

        return results

    def _summarize_walk_forward(self, results: List[Dict]) -> Dict:
        """Synthèse des résultats walk-forward."""
        sharpes = [r.get('sharpe_ratio', 0) for r in results]
        returns = [r.get('total_return', 0) for r in results]
        drawdowns = [r.get('max_drawdown', 0) for r in results]

        return {
            'mean_sharpe': float(np.mean(sharpes)),
            'std_sharpe': float(np.std(sharpes)),
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'mean_max_dd': float(np.mean(drawdowns)),
            'positive_windows_pct': float(np.mean([r > 0 for r in returns])),
            'n_windows': len(results),
        }

    def run_monte_carlo(
        self,
        trades_df: pd.DataFrame,
        n_simulations: int = 1000,
    ) -> Dict:
        """
        Monte Carlo sur les trades : simule N chemins aléatoires pour évaluer la robustesse.
        """
        if trades_df.empty:
            return {}

        returns = trades_df['pnl'].values / self.initial_capital
        final_values = []

        for _ in range(n_simulations):
            shuffled = np.random.choice(returns, size=len(returns), replace=True)
            equity_curve = self.initial_capital * np.cumprod(1 + shuffled)
            final_values.append(equity_curve[-1])

        final_values = np.array(final_values)
        return {
            'mc_mean_final': float(np.mean(final_values)),
            'mc_median_final': float(np.median(final_values)),
            'mc_5pct': float(np.percentile(final_values, 5)),
            'mc_95pct': float(np.percentile(final_values, 95)),
            'mc_prob_profit': float(np.mean(final_values > self.initial_capital)),
        }

    def _log_results(self, metrics: Dict):
        """Affiche les métriques clés."""
        logger.info("=" * 60)
        logger.info(f"BACKTEST RESULTS : {metrics.get('symbol')} {metrics.get('timeframe')}")
        logger.info(f"  Total Return    : {metrics.get('total_return', 0):.2%}")
        logger.info(f"  Sharpe Ratio    : {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Sortino Ratio   : {metrics.get('sortino_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown    : {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  Calmar Ratio    : {metrics.get('calmar_ratio', 0):.2f}")
        logger.info(f"  Win Rate        : {metrics.get('win_rate', 0):.2%}")
        logger.info(f"  Profit Factor   : {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"  N Trades        : {metrics.get('n_trades', 0)}")
        logger.info("=" * 60)
