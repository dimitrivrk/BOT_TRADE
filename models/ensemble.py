"""
Ensemble v3 : CryptoMamba + RL Ensemble (SAC+PPO+DDPG) + Signaux techniques.

Architecture état de l'art 2025 :
- CryptoMamba (SSM) remplace TFT pour la prédiction de prix
- Ensemble de 3 agents RL avec vote pondéré dynamique
- XGBoost feature selection pour des features optimales
- Adaptation dynamique des poids par régime de marché
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import yaml

from utils.logger import setup_logger

logger = setup_logger("models.ensemble")


class EnsemblePredictor:
    """
    Combine CryptoMamba, RL Ensemble et signaux techniques.

    Poids par défaut :
    - Mamba (prédiction prix) : 40%
    - RL Ensemble (décision) : 40%
    - Technique (confirmation) : 20%
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

        # Lazy loading des modèles
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

        # --- Signal CryptoMamba (prédiction de prix) ---
        if self.mamba is not None:
            try:
                if hasattr(self.mamba, 'model') and self.mamba.model is not None:
                    mamba_signal = self.mamba.predict(features_df, symbol)
                    signals['mamba'] = mamba_signal
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

        # --- Adapter les poids au régime ---
        weights = self._get_regime_weights(regime)

        # --- Combinaison ---
        final_signal = self._weighted_vote(signals, weights)

        # --- Position sizing basé sur la confiance ---
        final_signal['position_size'] = self._compute_position_size(
            final_signal['confidence'], regime,
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
        """Adapte les poids au régime de marché."""
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

    def _compute_position_size(self, confidence: float, regime: Optional[str]) -> float:
        """Position sizing basé sur confiance + régime."""
        base_size = confidence

        if regime == 'volatile':
            base_size *= 0.5
        elif regime == 'ranging':
            base_size *= 0.7

        kelly_fraction = 0.25
        position_size = base_size * kelly_fraction

        return float(np.clip(position_size, 0.0, 0.15))
