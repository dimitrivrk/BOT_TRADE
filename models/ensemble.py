"""
Ensemble des modèles : TFT + RL + signaux techniques.
Combine les prédictions par stacking ou vote pondéré.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import yaml

from models.tft_model import TFTPredictor
from models.rl_agent import RLTradingAgent
from utils.logger import setup_logger

logger = setup_logger("models.ensemble")


class EnsemblePredictor:
    """
    Combine TFT, RL et signaux techniques en un signal final.

    Méthodes disponibles :
    - 'weighted' : moyenne pondérée des signaux
    - 'stacking' : meta-modèle appris sur les prédictions individuelles
    - 'voting'   : vote majoritaire avec seuil de confiance
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["models"]["ensemble"]
        self.method = self.cfg.get("method", "weighted")
        self.tft_weight = self.cfg.get("tft_weight", 0.4)
        self.rl_weight = self.cfg.get("rl_weight", 0.4)
        self.technical_weight = self.cfg.get("technical_weight", 0.2)
        self.confidence_threshold = self.cfg.get("confidence_threshold", 0.60)

        # Modèles
        self.tft = TFTPredictor(config_path) if cfg["models"]["tft"]["enabled"] else None
        self.rl = RLTradingAgent(config_path) if cfg["models"]["rl"]["enabled"] else None

    def predict(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        symbol: str,
        regime: Optional[str] = None,
    ) -> Dict:
        """
        Génère le signal final combiné.

        Args:
            features_df: Features normalisées
            prices_df: OHLCV brut
            symbol: Paire tradée
            regime: Régime de marché détecté (optionnel, pour adapter les poids)

        Returns:
            {
                'direction': -1|0|1,
                'confidence': float (0-1),
                'position_size': float (0-1),  # fraction du capital
                'signal_details': dict          # décomposition par modèle
            }
        """
        signals = {}

        # --- Signal TFT ---
        if self.tft and self.tft.model is not None:
            try:
                tft_signal = self.tft.predict(features_df, symbol)
                signals['tft'] = tft_signal
            except Exception as e:
                logger.warning(f"TFT predict failed : {e}")
                signals['tft'] = {"direction": 0, "confidence": 0.0}

        # --- Signal RL ---
        if self.rl and self.rl.model is not None:
            try:
                rl_signal = self.rl.predict(features_df, prices_df)
                signals['rl'] = rl_signal
            except Exception as e:
                logger.warning(f"RL predict failed : {e}")
                signals['rl'] = {"direction": 0, "confidence": 0.0}

        # --- Signal technique ---
        tech_signal = self._compute_technical_signal(features_df)
        signals['technical'] = tech_signal

        # --- Adapter les poids au régime ---
        weights = self._get_regime_weights(regime)

        # --- Combinaison ---
        if self.method == "weighted":
            final_signal = self._weighted_vote(signals, weights)
        elif self.method == "voting":
            final_signal = self._majority_vote(signals)
        else:
            final_signal = self._weighted_vote(signals, weights)

        # --- Position sizing basé sur la confiance ---
        final_signal['position_size'] = self._compute_position_size(
            final_signal['confidence'],
            regime,
        )
        final_signal['signal_details'] = signals
        final_signal['regime'] = regime

        logger.info(
            f"{symbol} Signal : dir={final_signal['direction']:+d} | "
            f"conf={final_signal['confidence']:.2%} | "
            f"size={final_signal['position_size']:.2%}"
        )

        return final_signal

    def _compute_technical_signal(self, features_df: pd.DataFrame) -> Dict:
        """
        Signal technique basé sur les features calculées.
        Utilise un ensemble de règles de trading classiques.
        """
        if features_df.empty:
            return {"direction": 0, "confidence": 0.0}

        last = features_df.iloc[-1]
        score = 0.0
        n_signals = 0

        # RSI
        if 'rsi_14' in last.index:
            rsi = last['rsi_14']  # déjà normalisé 0-1
            if rsi < 0.3:         # oversold
                score += 1.0
            elif rsi > 0.7:       # overbought
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
            if last['sq_on'] == 0:  # squeeze vient de se relâcher
                score += np.sign(last['sq_momentum']) * 0.5
                n_signals += 0.5

        if n_signals == 0:
            return {"direction": 0, "confidence": 0.0}

        normalized_score = score / n_signals  # [-1, 1]
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

        for model_name, signal in signals.items():
            w = weights.get(model_name, 0.0)
            if w == 0 or signal.get('confidence', 0) == 0:
                continue

            # Score pondéré par la confiance du modèle
            model_score = signal['direction'] * signal['confidence']
            total_score += w * model_score
            weighted_confidence += w * signal['confidence']
            total_weight += w

        if total_weight == 0:
            return {"direction": 0, "confidence": 0.0}

        final_score = total_score / total_weight
        final_confidence = weighted_confidence / total_weight

        direction = 1 if final_score > 0.2 else (-1 if final_score < -0.2 else 0)

        # Bloquer si confiance insuffisante
        if final_confidence < self.confidence_threshold:
            direction = 0

        return {
            "direction": direction,
            "confidence": float(final_confidence),
            "raw_score": float(final_score),
        }

    def _majority_vote(self, signals: Dict) -> Dict:
        """Vote majoritaire simple."""
        votes = [s['direction'] for s in signals.values() if s.get('confidence', 0) > 0.3]
        confidences = [s['confidence'] for s in signals.values() if s.get('confidence', 0) > 0.3]

        if not votes:
            return {"direction": 0, "confidence": 0.0}

        vote_sum = sum(votes)
        direction = 1 if vote_sum > 0 else (-1 if vote_sum < 0 else 0)

        # Confiance proportionnelle à l'unanimité
        unanimity = abs(vote_sum) / len(votes)
        avg_confidence = np.mean(confidences)
        final_confidence = unanimity * avg_confidence

        if final_confidence < self.confidence_threshold:
            direction = 0

        return {
            "direction": direction,
            "confidence": float(final_confidence),
        }

    def _get_regime_weights(self, regime: Optional[str]) -> Dict:
        """
        Adapte les poids au régime de marché.
        Le TFT performe mieux en tendance, le RL en ranging.
        """
        base = {
            'tft': self.tft_weight,
            'rl': self.rl_weight,
            'technical': self.technical_weight,
        }

        if regime == 'trending_bull' or regime == 'trending_bear':
            # En tendance → favoriser TFT (meilleure capture des tendances)
            base['tft'] = min(base['tft'] * 1.3, 0.6)
            base['technical'] = max(base['technical'] * 0.7, 0.1)

        elif regime == 'ranging':
            # En range → favoriser RL et signaux techniques (RSI oversold/overbought)
            base['rl'] = min(base['rl'] * 1.3, 0.6)
            base['technical'] = min(base['technical'] * 1.3, 0.4)

        elif regime == 'volatile':
            # Volatile → réduire toutes les positions
            base = {k: v * 0.5 for k, v in base.items()}

        # Normaliser
        total = sum(base.values())
        return {k: v / total for k, v in base.items()}

    def _compute_position_size(
        self,
        confidence: float,
        regime: Optional[str],
    ) -> float:
        """
        Calcule la taille de position en % du capital.
        Basé sur Kelly fractionné × confiance × ajustement régime.
        """
        # Base : proportionnel à la confiance
        base_size = confidence

        # Réduire en régime volatile
        if regime == 'volatile':
            base_size *= 0.5
        elif regime == 'ranging':
            base_size *= 0.7

        # Kelly fractionné (25% du Kelly full)
        kelly_fraction = 0.25
        position_size = base_size * kelly_fraction

        return float(np.clip(position_size, 0.0, 0.15))  # max 15% du capital
