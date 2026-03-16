"""
Ensemble v4 : CryptoMamba + RL Ensemble (SAC+PPO+DDPG) + Signaux techniques.

Architecture etat de l'art 2025 :
- CryptoMamba (SSM) remplace TFT pour la prediction de prix
- Ensemble de 3 agents RL avec vote pondere dynamique
- XGBoost feature selection pour des features optimales
- Meta-learner adaptatif : poids dynamiques bases sur la performance recente
  par regime de marche (au lieu de poids fixes)
- Multi-horizon analysis : exploite les predictions Mamba 6h pour SL/TP
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque
import yaml

from utils.logger import setup_logger

logger = setup_logger("models.ensemble")


class MetaLearner:
    """
    Meta-learner adaptatif pour les poids de l'ensemble.

    Au lieu de poids fixes, le meta-learner suit la performance recente de
    chaque modele par regime de marche et ajuste les poids dynamiquement.

    Principe: Exponential Moving Average des predictions correctes par modele,
    segmente par regime. Le modele qui a le plus souvent raison recemment
    dans le regime courant recoit le plus de poids.

    Reference: FinRL Ensemble (Columbia AI4Finance, 2024)
    """

    def __init__(self, model_names: list, decay: float = 0.95, min_weight: float = 0.05):
        self.model_names = model_names
        self.decay = decay  # EMA decay factor
        self.min_weight = min_weight
        self.window = 50  # nombre de predictions a garder

        # Performance tracking par regime
        self.regime_scores: Dict[str, Dict[str, deque]] = {}
        for regime in ['trending_bull', 'trending_bear', 'ranging', 'volatile', 'unknown']:
            self.regime_scores[regime] = {
                name: deque(maxlen=self.window) for name in model_names
            }

        # Default weights (point de depart)
        self._default_weights = {name: 1.0 / len(model_names) for name in model_names}

    def record_prediction(self, model_name: str, direction: int, actual_return: float, regime: str):
        """
        Enregistre si la prediction d'un modele etait correcte.

        Args:
            model_name: nom du modele ('mamba', 'rl', 'technical')
            direction: direction predite (-1, 0, 1)
            actual_return: retour reel de la bougie suivante
            regime: regime de marche courant
        """
        if model_name not in self.model_names:
            return
        if regime not in self.regime_scores:
            regime = 'unknown'

        # Score: 1 si bonne direction, 0 sinon, 0.5 si neutre
        if direction == 0:
            score = 0.3  # penalite douce pour l'abstention
        elif (direction > 0 and actual_return > 0) or (direction < 0 and actual_return < 0):
            score = 1.0  # bonne prediction
        else:
            score = 0.0  # mauvaise prediction

        self.regime_scores[regime][model_name].append(score)

    def get_weights(self, regime: str) -> Dict[str, float]:
        """
        Retourne les poids optimaux pour le regime courant,
        bases sur la performance EMA recente.
        """
        if regime not in self.regime_scores:
            regime = 'unknown'

        scores = {}
        for name in self.model_names:
            history = list(self.regime_scores[regime][name])
            if len(history) < 5:
                # Pas assez d'historique -> poids par defaut
                scores[name] = self._default_weights[name]
            else:
                # EMA des scores recents
                ema = history[0]
                for s in history[1:]:
                    ema = self.decay * ema + (1 - self.decay) * s
                scores[name] = max(ema, self.min_weight)

        # Normaliser
        total = sum(scores.values())
        weights = {k: v / total for k, v in scores.items()}

        return weights

    def get_stats(self) -> Dict:
        """Retourne les stats du meta-learner pour le debug."""
        stats = {}
        for regime, models in self.regime_scores.items():
            regime_stats = {}
            for name, history in models.items():
                if len(history) > 0:
                    regime_stats[name] = {
                        'n': len(history),
                        'avg': float(np.mean(list(history))),
                        'recent': float(list(history)[-1]) if history else 0,
                    }
            if regime_stats:
                stats[regime] = regime_stats
        return stats


class EnsemblePredictor:
    """
    Combine CryptoMamba, RL Ensemble et signaux techniques.

    v4: Poids dynamiques via MetaLearner (au lieu de fixes).
    Les poids de base sont un fallback, le meta-learner les overwrite
    des qu'il a assez de donnees (>5 predictions par regime).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["models"]["ensemble"]
        self.method = self.cfg.get("method", "weighted")
        self.mamba_weight = self.cfg.get("mamba_weight", 0.40)
        self.rl_weight = self.cfg.get("rl_weight", 0.40)
        self.technical_weight = self.cfg.get("technical_weight", 0.20)
        self.confidence_threshold = self.cfg.get("confidence_threshold", 0.25)

        # Meta-learner adaptatif
        self.meta_learner = MetaLearner(
            model_names=['mamba', 'rl', 'technical'],
            decay=0.95,
            min_weight=0.05,
        )
        self._last_predictions: Dict[str, Dict] = {}  # pour le feedback au meta-learner

        # Lazy loading des modeles
        self.mamba = None
        self.rl = None
        self._models_loaded = False

        # Stocker le config path pour lazy loading
        self._config_path = config_path
        self._cfg = cfg

    def _load_models(self):
        """Charge les modèles à la demande (lazy loading)."""
        if self._models_loaded:
            return

        # CryptoMamba
        if self._cfg["models"].get("mamba", {}).get("enabled", True):
            try:
                from models.crypto_mamba import MambaPredictor
                self.mamba = MambaPredictor(self._config_path)
                logger.info("CryptoMamba initialisé")
            except Exception as e:
                logger.warning(f"CryptoMamba non disponible : {e}")

        # RL Ensemble
        if self._cfg["models"]["rl"]["enabled"]:
            try:
                from models.rl_agent import RLTradingAgent
                self.rl = RLTradingAgent(self._config_path)
                logger.info("RL Ensemble initialisé")
            except Exception as e:
                logger.warning(f"RL Ensemble non disponible : {e}")

        # Fallback TFT si Mamba pas dispo
        if self.mamba is None and self._cfg["models"].get("tft", {}).get("enabled", False):
            try:
                from models.tft_model import TFTPredictor
                self.mamba = TFTPredictor(self._config_path)
                logger.info("Fallback TFT chargé (Mamba non disponible)")
            except Exception as e:
                logger.warning(f"TFT fallback non disponible : {e}")

        self._models_loaded = True

    def predict(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        symbol: str,
        regime: Optional[str] = None,
    ) -> Dict:
        """
        Génère le signal final combiné.

        Returns:
            {
                'direction': -1|0|1,
                'confidence': float (0-1),
                'position_size': float (0-1),
                'signal_details': dict
            }
        """
        self._load_models()
        signals = {}

        # --- Signal CryptoMamba (prediction de prix multi-horizon) ---
        if self.mamba is not None:
            try:
                if hasattr(self.mamba, 'model') and self.mamba.model is not None:
                    mamba_signal = self.mamba.predict(features_df, symbol)
                    signals['mamba'] = mamba_signal

                    # Exploiter les predictions multi-horizon pour la strategie
                    predicted_returns = mamba_signal.get('predicted_returns', [])
                    if len(predicted_returns) >= 3:
                        mh = self._analyze_multi_horizon(predicted_returns, mamba_signal['direction'])
                        mamba_signal['multi_horizon'] = mh
            except Exception as e:
                logger.warning(f"Mamba predict failed : {e}")
                signals['mamba'] = {"direction": 0, "confidence": 0.0}

        # --- Signal RL Ensemble ---
        if self.rl is not None:
            try:
                rl_signal = self.rl.predict(features_df, prices_df)
                signals['rl'] = rl_signal
            except Exception as e:
                logger.warning(f"RL predict failed : {e}")
                signals['rl'] = {"direction": 0, "confidence": 0.0}

        # --- Signal technique ---
        tech_signal = self._compute_technical_signal(features_df)
        signals['technical'] = tech_signal

        # --- Feedback meta-learner avec le return de la bougie precedente ---
        if not prices_df.empty and len(prices_df) >= 2:
            prev_return = float(np.log(prices_df['close'].iloc[-1] / prices_df['close'].iloc[-2]))
            self.feedback(prev_return, regime)

        # Stocker les predictions actuelles pour le prochain feedback
        self._last_predictions = {
            model: {'direction': sig.get('direction', 0)}
            for model, sig in signals.items()
        }

        # --- Adapter les poids au regime (meta-learner ou heuristique) ---
        weights = self._get_regime_weights(regime)

        # --- Combinaison ---
        final_signal = self._weighted_vote(signals, weights)

        # --- Position sizing base sur la confiance ---
        final_signal['position_size'] = self._compute_position_size(
            final_signal['confidence'], regime,
        )
        final_signal['signal_details'] = signals
        final_signal['regime'] = regime

        # --- Multi-horizon intelligence (si Mamba a des predictions) ---
        mamba_sig = signals.get('mamba', {})
        if 'multi_horizon' in mamba_sig:
            mh = mamba_sig['multi_horizon']
            final_signal['multi_horizon'] = mh
            # Ajuster position size selon l'accord multi-horizon
            if mh['horizon_agreement'] > 0.8:
                final_signal['position_size'] *= 1.2  # boost si forte conviction
            elif mh['horizon_agreement'] < 0.4:
                final_signal['position_size'] *= 0.7  # reduire si desaccord
            final_signal['position_size'] = float(np.clip(final_signal['position_size'], 0, 0.15))

        # Raw scores pour Discord / logging
        final_signal['raw_scores'] = {
            'mamba': mamba_sig.get('confidence', 0) * mamba_sig.get('direction', 0),
            'rl': signals.get('rl', {}).get('confidence', 0) * signals.get('rl', {}).get('direction', 0),
            'technical': signals.get('technical', {}).get('raw_score', 0),
        }

        logger.info(
            f"{symbol} Signal : dir={final_signal['direction']:+d} | "
            f"conf={final_signal['confidence']:.2%} | "
            f"size={final_signal['position_size']:.2%}"
        )

        return final_signal

    def _compute_technical_signal(self, features_df: pd.DataFrame) -> Dict:
        """Signal technique basé sur les features calculées."""
        if features_df.empty:
            return {"direction": 0, "confidence": 0.0}

        last = features_df.iloc[-1]
        score = 0.0
        n_signals = 0

        # RSI
        if 'rsi_14' in last.index:
            rsi = last['rsi_14']
            if rsi < 0.3:
                score += 1.0
            elif rsi > 0.7:
                score -= 1.0
            n_signals += 1

        # MACD
        if 'macd_diff' in last.index:
            score += np.sign(last['macd_diff'])
            n_signals += 1

        # EMA trend
        if 'ema_9_21_cross' in last.index:
            score += np.sign(last['ema_9_21_cross'])
            n_signals += 1
        if 'ema_21_50_cross' in last.index:
            score += np.sign(last['ema_21_50_cross'])
            n_signals += 1

        # SuperTrend
        if 'supertrend_dir' in last.index:
            score += float(last['supertrend_dir'])
            n_signals += 1

        # Volume confirmation
        if 'vol_ratio_20' in last.index and 'ret_1' in last.index:
            vol_confirmation = np.sign(last['ret_1']) * min(last['vol_ratio_20'] - 1, 1.0)
            score += vol_confirmation * 0.5
            n_signals += 0.5

        # Squeeze Momentum
        if 'sq_momentum' in last.index and 'sq_on' in last.index:
            if last['sq_on'] == 0:
                score += np.sign(last['sq_momentum']) * 0.5
                n_signals += 0.5

        # --- Lower Timeframe (15m) micro-structure confirmation ---
        # Ces features ne changent PAS la direction mais ajustent la confiance
        ltf_confirmation = 0.0
        ltf_signals = 0

        # Trend consistency 15m : 4/4 bougies 15m dans la même direction = forte conviction
        if 'ltf_trend_consistency' in last.index:
            tc = last['ltf_trend_consistency']
            if not np.isnan(tc):
                # tc=1.0 → 4/4 bullish, tc=0.0 → 4/4 bearish, tc=0.5 → neutre
                ltf_dir = 1.0 if tc > 0.7 else (-1.0 if tc < 0.3 else 0.0)
                ltf_confirmation += ltf_dir
                ltf_signals += 1

        # RSI 15m ultra-court terme (confirme le momentum)
        if 'ltf_rsi_8_15m' in last.index:
            rsi_ltf = last['ltf_rsi_8_15m']
            if not np.isnan(rsi_ltf):
                if rsi_ltf < 0.25:
                    ltf_confirmation += 1.0  # oversold → bullish
                elif rsi_ltf > 0.75:
                    ltf_confirmation -= 1.0  # overbought → bearish
                ltf_signals += 1

        # Breakout score 15m (cassure de range intra-heure)
        if 'ltf_breakout_score' in last.index:
            bs = last['ltf_breakout_score']
            if not np.isnan(bs) and abs(bs) > 0.001:
                ltf_confirmation += np.sign(bs) * 0.5
                ltf_signals += 0.5

        # Volume spike 15m (confirme un vrai mouvement)
        if 'ltf_vol_spike_ltf' in last.index:
            vs = last['ltf_vol_spike_ltf']
            if not np.isnan(vs) and vs > 2.0:
                # Fort volume = confirme la direction du ret_4_15m
                if 'ltf_ret_4_15m' in last.index:
                    ret = last['ltf_ret_4_15m']
                    if not np.isnan(ret):
                        ltf_confirmation += np.sign(ret) * 0.5
                        ltf_signals += 0.5

        # Intégrer la confirmation LTF avec un poids modéré
        if ltf_signals > 0:
            ltf_score = ltf_confirmation / ltf_signals
            score += ltf_score * 0.7  # poids 0.7 pour le LTF (ne domine pas)
            n_signals += 0.7

        if n_signals == 0:
            return {"direction": 0, "confidence": 0.0}

        normalized_score = score / n_signals
        direction = 1 if normalized_score > 0.2 else (-1 if normalized_score < -0.2 else 0)
        confidence = min(abs(normalized_score), 1.0)

        return {
            "direction": direction,
            "confidence": confidence,
            "raw_score": float(normalized_score),
            "model": "technical",
        }

    def _weighted_vote(self, signals: Dict, weights: Dict) -> Dict:
        """Combinaison par moyenne pondérée."""
        total_score = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0

        # Mapping des noms de modèles aux poids
        weight_map = {
            'mamba': weights.get('mamba', 0.4),
            'tft': weights.get('mamba', 0.4),  # TFT utilise le même poids que Mamba
            'rl': weights.get('rl', 0.4),
            'technical': weights.get('technical', 0.2),
        }

        for model_name, signal in signals.items():
            w = weight_map.get(model_name, 0.0)
            if w == 0 or signal.get('confidence', 0) == 0:
                continue

            model_score = signal['direction'] * signal['confidence']
            total_score += w * model_score
            weighted_confidence += w * signal['confidence']
            total_weight += w

        if total_weight == 0:
            return {"direction": 0, "confidence": 0.0}

        final_score = total_score / total_weight
        final_confidence = weighted_confidence / total_weight

        direction = 1 if final_score > 0.15 else (-1 if final_score < -0.15 else 0)

        if final_confidence < self.confidence_threshold:
            direction = 0

        return {
            "direction": direction,
            "confidence": float(final_confidence),
            "raw_score": float(final_score),
        }

    def _get_regime_weights(self, regime: Optional[str]) -> Dict:
        """
        Poids adaptatifs via le meta-learner.
        Le meta-learner utilise l'historique des predictions correctes par regime.
        Fallback sur les poids heuristiques si pas assez de donnees.
        """
        # Essayer le meta-learner d'abord
        if regime:
            meta_weights = self.meta_learner.get_weights(regime)
            # Verifier qu'on a assez de donnees
            regime_data = self.meta_learner.regime_scores.get(regime, {})
            has_enough = all(
                len(regime_data.get(name, [])) >= 5
                for name in ['mamba', 'rl', 'technical']
            )
            if has_enough:
                logger.debug(f"Meta-learner weights ({regime}): {meta_weights}")
                return meta_weights

        # Fallback: poids heuristiques par regime
        base = {
            'mamba': self.mamba_weight,
            'rl': self.rl_weight,
            'technical': self.technical_weight,
        }

        if regime == 'trending_bull' or regime == 'trending_bear':
            base['mamba'] = min(base['mamba'] * 1.3, 0.55)
            base['technical'] = max(base['technical'] * 0.7, 0.1)
        elif regime == 'ranging':
            base['rl'] = min(base['rl'] * 1.3, 0.55)
            base['technical'] = min(base['technical'] * 1.3, 0.35)
        elif regime == 'volatile':
            base = {k: v * 0.6 for k, v in base.items()}

        total = sum(base.values())
        return {k: v / total for k, v in base.items()}

    def feedback(self, actual_return: float, regime: Optional[str] = None):
        """
        Feedback au meta-learner apres que la bougie soit fermee.
        Compare les predictions de la derniere bougie au retour reel.

        Appeler cette methode dans on_new_candle avec le retour de la bougie precedente.
        """
        if not self._last_predictions:
            return
        if regime is None:
            regime = 'unknown'

        for model_name, pred in self._last_predictions.items():
            direction = pred.get('direction', 0)
            self.meta_learner.record_prediction(model_name, direction, actual_return, regime)

    def _compute_position_size(self, confidence: float, regime: Optional[str]) -> float:
        """Position sizing base sur confiance + regime."""
        base_size = confidence

        if regime == 'volatile':
            base_size *= 0.5
        elif regime == 'ranging':
            base_size *= 0.7

        kelly_fraction = 0.25
        position_size = base_size * kelly_fraction

        return float(np.clip(position_size, 0.0, 0.15))

    def _analyze_multi_horizon(self, predicted_returns: list, direction: int) -> Dict:
        """
        Analyse les predictions multi-horizon de Mamba pour determiner:
        1. La duree estimee du move (combien d'heures dans la meme direction)
        2. Le return cumule attendu (pour calibrer le TP)
        3. Le point de retournement estime (pour le SL dynamique)
        4. La force du trend (accelere ou decelere)

        Args:
            predicted_returns: liste de 6 returns predits (h+1 a h+6)
            direction: 1=long, -1=short

        Returns:
            dict avec les metriques multi-horizon
        """
        preds = np.array(predicted_returns)

        # 1. Duree estimee du move (nb d'heures dans la meme direction)
        if direction == 1:
            same_dir = np.cumsum(preds > 0)
            move_duration = 0
            for i, p in enumerate(preds):
                if p > 0:
                    move_duration = i + 1
                else:
                    break
        elif direction == -1:
            move_duration = 0
            for i, p in enumerate(preds):
                if p < 0:
                    move_duration = i + 1
                else:
                    break
        else:
            move_duration = 0

        # 2. Return cumule attendu sur la duree du move
        if move_duration > 0:
            cumulative_return = float(np.sum(preds[:move_duration]))
        else:
            cumulative_return = float(np.sum(preds))

        # 3. Max adverse excursion predit (pire drawdown attendu)
        cum_returns = np.cumsum(preds) * direction  # positif = favorable
        max_adverse = float(np.min(cum_returns)) if len(cum_returns) > 0 else 0.0

        # 4. Force du trend: compare la 1ere moitie vs 2eme moitie
        half = len(preds) // 2
        first_half_avg = float(np.mean(np.abs(preds[:half])))
        second_half_avg = float(np.mean(np.abs(preds[half:])))
        trend_momentum = first_half_avg - second_half_avg  # >0 = decelere, <0 = accelere

        # 5. Confiance multi-horizon (% de predictions dans la meme direction)
        if direction != 0:
            agreement = float(np.mean(np.sign(preds) == direction))
        else:
            agreement = 0.0

        # 6. Multiplicateur de TP suggere (si le move dure longtemps, TP plus loin)
        tp_multiplier = 1.0 + max(0, (move_duration - 2)) * 0.25  # +25% par heure > 2

        # 7. Multiplicateur de SL suggere (si max adverse est faible, SL plus serre)
        if max_adverse < -0.001:
            sl_multiplier = 0.8  # serrer le SL car le modele predit un drawdown
        elif max_adverse > 0:
            sl_multiplier = 1.2  # elargir car pas de drawdown attendu
        else:
            sl_multiplier = 1.0

        result = {
            'move_duration_hours': int(move_duration),
            'cumulative_return': cumulative_return,
            'max_adverse_excursion': max_adverse,
            'trend_momentum': trend_momentum,  # >0=decelerant, <0=accelerant
            'horizon_agreement': agreement,
            'tp_multiplier': float(np.clip(tp_multiplier, 0.8, 2.5)),
            'sl_multiplier': float(np.clip(sl_multiplier, 0.7, 1.5)),
        }

        logger.debug(
            f"Multi-horizon: duree={move_duration}h, cum_ret={cumulative_return:.4f}, "
            f"agreement={agreement:.0%}, tp_mult={tp_multiplier:.2f}, sl_mult={sl_multiplier:.2f}"
        )

        return result
