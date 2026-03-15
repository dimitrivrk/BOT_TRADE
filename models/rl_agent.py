"""
Agent de Reinforcement Learning v2 pour le trading crypto.
Améliorations vs v1 :
  - SAC (Soft Actor-Critic) par défaut au lieu de PPO → plus stable, off-policy
  - Differential Sharpe Ratio comme reward → signal plus lisse
  - Pénalités : drawdown + turnover excessif + inactivité
  - target_kl sur PPO en fallback
  - Meilleure normalisation des observations
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import yaml
import pickle

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise

import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger("models.rl_agent")


# =============================================================================
# ENVIRONNEMENT DE TRADING v2
# =============================================================================

class CryptoTradingEnv(gym.Env):
    """
    Environnement Gym v2 pour le trading crypto.

    Améliorations :
    - Differential Sharpe Ratio (Moody & Saffell 1998)
    - Pénalité de turnover (freine le churning)
    - Pénalité drawdown progressive
    - Bonus pour inactivité intelligente (flat quand incertain)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        config: dict,
        mode: str = "train",
    ):
        super().__init__()
        self.features = features_df.values.astype(np.float32)
        self.prices = prices_df['close'].values.astype(np.float32)
        self.returns = np.diff(np.log(self.prices))  # log returns

        self.n_features = self.features.shape[1]
        self.lookback = 50
        self.mode = mode

        # Config
        self.initial_balance = config.get("initial_balance", 10000)
        self.transaction_cost = config.get("transaction_cost", 0.001)
        self.reward_type = config.get("reward_type", "differential_sharpe")

        # Differential Sharpe Ratio state (EMA)
        self.eta = 0.01  # decay factor pour DSR
        self.A_prev = 0.0  # EMA des returns
        self.B_prev = 0.0  # EMA des returns^2

        # Espaces Gym
        obs_size = self.lookback * self.n_features
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomiser le point de départ en mode train (meilleure exploration)
        if self.mode == "train" and len(self.prices) > self.lookback + 500:
            max_start = len(self.prices) - 500
            self.current_step = self.np_random.integers(self.lookback, max_start)
        else:
            self.current_step = self.lookback

        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.returns_history = []
        self.position_history = []
        self.portfolio_values = [self.initial_balance]
        self.total_fees = 0.0
        self.peak_balance = self.initial_balance

        # Reset DSR state
        self.A_prev = 0.0
        self.B_prev = 0.0

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.lookback
        obs = self.features[start:self.current_step]
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return obs.flatten().astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        new_position = float(np.clip(action[0], -1.0, 1.0))

        # Coût de transaction
        position_change = abs(new_position - self.position)
        fee = position_change * self.transaction_cost * self.balance
        self.total_fees += fee
        self.balance -= fee

        # Retour de marché
        if self.current_step < len(self.returns):
            market_return = self.returns[self.current_step - 1]
        else:
            market_return = 0.0

        # PnL
        pnl = self.position * market_return * self.balance
        self.balance += pnl

        # Track peak pour drawdown
        self.peak_balance = max(self.peak_balance, self.balance)

        # Update position
        self.position = new_position

        # Historique
        step_return = pnl / (self.portfolio_values[-1] + 1e-8)
        self.returns_history.append(step_return)
        self.position_history.append(new_position)
        self.portfolio_values.append(self.balance)

        # Reward
        reward = self._compute_reward(position_change)

        # Avancer
        self.current_step += 1
        terminated = (
            self.current_step >= len(self.prices) - 1
            or self.balance < self.initial_balance * 0.5
        )
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info = {
            "balance": self.balance,
            "position": self.position,
            "total_return": (self.balance - self.initial_balance) / self.initial_balance,
            "total_fees": self.total_fees,
            "drawdown": (self.balance - self.peak_balance) / (self.peak_balance + 1e-8),
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, position_change: float) -> float:
        """
        Reward multi-composantes :
        1. Differential Sharpe Ratio (signal principal)
        2. Pénalité de turnover
        3. Pénalité de drawdown progressive
        """
        if len(self.returns_history) < 2:
            return 0.0

        R_t = self.returns_history[-1]  # return du step courant
        reward = 0.0

        # === 1. Differential Sharpe Ratio (Moody & Saffell) ===
        # DSR = dSharpe/dt, mesure l'amélioration marginale du Sharpe
        A_new = self.A_prev + self.eta * (R_t - self.A_prev)
        B_new = self.B_prev + self.eta * (R_t**2 - self.B_prev)

        denom = (B_new - A_new**2) ** 1.5 + 1e-12
        if abs(denom) > 1e-10:
            dsr = (B_new * (R_t - A_new) - 0.5 * A_new * (R_t**2 - B_new)) / denom
        else:
            dsr = R_t  # fallback au PnL simple

        self.A_prev = A_new
        self.B_prev = B_new

        # Normaliser le DSR pour qu'il soit dans une plage raisonnable
        reward += float(np.clip(dsr * 100, -5.0, 5.0))

        # === 2. Pénalité de turnover (freine le churning) ===
        turnover_penalty = -0.1 * position_change
        reward += turnover_penalty

        # === 3. Pénalité de drawdown progressive ===
        if self.peak_balance > 0:
            dd = (self.balance - self.peak_balance) / self.peak_balance
            if dd < -0.05:
                reward += dd * 3.0  # pénalité proportionnelle au DD
            if dd < -0.15:
                reward -= 2.0  # pénalité forte si DD > 15%

        # === 4. Petit bonus PnL pour garder le signal informatif ===
        reward += R_t * 10.0  # scale le return brut

        return float(np.clip(reward, -10.0, 10.0))

    def render(self, mode="human"):
        dd = (self.balance - self.peak_balance) / (self.peak_balance + 1e-8)
        print(
            f"Step {self.current_step} | Balance: {self.balance:.2f} | "
            f"Position: {self.position:.2f} | "
            f"DD: {dd:.2%} | "
            f"Return: {(self.balance - self.initial_balance) / self.initial_balance:.2%}"
        )

    def get_performance_metrics(self) -> dict:
        returns = np.array(self.returns_history)
        portfolio = np.array(self.portfolio_values)

        if len(returns) < 2:
            return {}

        total_return = (portfolio[-1] - portfolio[0]) / portfolio[0]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)
        max_dd = np.min(portfolio / np.maximum.accumulate(portfolio) - 1)
        downside = np.std(returns[returns < 0]) + 1e-8
        sortino = np.mean(returns) / downside * np.sqrt(252 * 24)

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(total_return / abs(max_dd + 1e-8)),
            "total_fees": float(self.total_fees),
        }


# =============================================================================
# FEATURE EXTRACTOR (Transformer 1D)
# =============================================================================

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor Transformer 1D pour capturer les dépendances temporelles.
    """

    def __init__(self, observation_space: spaces.Box, lookback: int = 50, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        n_total = observation_space.shape[0]
        self.lookback = lookback
        self.n_features = n_total // lookback

        self.input_proj = nn.Linear(self.n_features, 128)
        self.layer_norm = nn.LayerNorm(128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.view(-1, self.lookback, self.n_features)
        x = self.input_proj(x)
        x = self.layer_norm(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # dernier token
        return self.output_proj(x)


# =============================================================================
# AGENT RL v2
# =============================================================================

class RLTradingAgent:
    """
    Agent RL v2 : SAC par défaut, PPO en fallback.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["models"]["rl"]
        self.env_cfg = self.cfg.get("env", {})
        self.checkpoint_dir = Path(self.cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.vec_env: Optional[VecNormalize] = None
        self.algo_name = self.cfg.get("algorithm", "SAC")

    def _make_env(self, features_df, prices_df, mode="train"):
        def _init():
            env = CryptoTradingEnv(features_df, prices_df, self.env_cfg, mode=mode)
            env = Monitor(env)
            return env
        return _init

    def train(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        val_features_df: Optional[pd.DataFrame] = None,
        val_prices_df: Optional[pd.DataFrame] = None,
    ):
        """Entraîne l'agent RL."""
        # Aligner features et prices
        common_idx = features_df.index.intersection(prices_df.index)
        features_df = features_df.loc[common_idx]
        prices_df = prices_df.loc[common_idx]
        logger.info(f"RL données alignées : {len(features_df)} lignes, {features_df.shape[1]} features")

        # Split train/val (80/20)
        split = int(len(features_df) * 0.8)
        train_feat = features_df.iloc[:split]
        train_prices = prices_df.iloc[:split]
        val_feat = val_features_df if val_features_df is not None else features_df.iloc[split:]
        val_prices = val_prices_df if val_prices_df is not None else prices_df.iloc[split:]

        # Environnement d'entraînement
        env = DummyVecEnv([self._make_env(train_feat, train_prices)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)
        self.vec_env = env

        # Feature extractor
        lookback = 50
        policy_kwargs = dict(
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(lookback=lookback, features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=nn.ReLU,
        )

        # Validation env
        val_env = DummyVecEnv([self._make_env(val_feat, val_prices, mode="val")])
        val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, training=False)

        # Callbacks
        callbacks = [
            EvalCallback(
                val_env,
                best_model_save_path=str(self.checkpoint_dir),
                log_path=str(self.checkpoint_dir / "logs"),
                eval_freq=5000,
                n_eval_episodes=3,  # 3 épisodes pour eval plus stable
                deterministic=True,
            ),
            CheckpointCallback(
                save_freq=25000,
                save_path=str(self.checkpoint_dir),
                name_prefix="rl_checkpoint",
            ),
        ]

        algo = self.cfg.get("algorithm", "SAC")
        self.algo_name = algo
        lr = float(self.cfg["learning_rate"])

        if algo == "SAC":
            # SAC : off-policy, plus stable, meilleur pour le trading
            # net_arch format différent pour SAC
            sac_policy_kwargs = dict(
                features_extractor_class=TransformerFeaturesExtractor,
                features_extractor_kwargs=dict(lookback=lookback, features_dim=256),
                net_arch=[256, 128],
                activation_fn=nn.ReLU,
            )

            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=lr,
                buffer_size=int(self.cfg.get("buffer_size", 500_000)),
                learning_starts=int(self.cfg.get("learning_starts", 5000)),
                batch_size=self.cfg.get("batch_size", 512),
                tau=float(self.cfg.get("tau", 0.005)),
                gamma=float(self.cfg.get("gamma", 0.99)),
                train_freq=4,         # update tous les 4 steps (au lieu de 1)
                gradient_steps=4,     # 4 gradient steps par update
                ent_coef="auto",
                target_entropy="auto",
                policy_kwargs=sac_policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/rl",
                device="auto",
            )

        elif algo == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=lr,
                n_steps=self.cfg.get("n_steps", 1024),
                batch_size=self.cfg.get("batch_size", 256),
                n_epochs=self.cfg.get("n_epochs", 10),
                gamma=float(self.cfg.get("gamma", 0.99)),
                gae_lambda=float(self.cfg.get("gae_lambda", 0.95)),
                clip_range=float(self.cfg.get("clip_range", 0.2)),
                ent_coef=float(self.cfg.get("ent_coef", 0.01)),
                target_kl=float(self.cfg.get("target_kl", 0.02)),
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/rl",
                device="auto",
            )

        total_timesteps = self.cfg.get("total_timesteps", 500_000)
        logger.info(f"Entraînement RL ({algo}) : {total_timesteps:,} timesteps | lr={lr}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )

        # Sauvegarder
        env.save(str(self.checkpoint_dir / "vec_normalize.pkl"))
        self.model.save(str(self.checkpoint_dir / "rl_final"))
        logger.info("Agent RL entraîné et sauvegardé !")

        return self.model

    def predict(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Génère un signal de trading."""
        if self.model is None:
            self.load()

        # Ajuster features pour matcher le modèle entraîné
        expected_obs_size = self.model.observation_space.shape[0]
        lookback = 50
        expected_n_features = expected_obs_size // lookback
        if features_df.shape[1] > expected_n_features:
            features_df = features_df.iloc[:, :expected_n_features]
        elif features_df.shape[1] < expected_n_features:
            logger.warning(f"RL predict: {features_df.shape[1]} features < {expected_n_features} attendues")

        # Env temporaire pour l'inférence
        env = CryptoTradingEnv(
            features_df.tail(100 + 50),
            prices_df.tail(100 + 50),
            self.env_cfg,
            mode="inference",
        )
        obs, _ = env.reset()

        # Avancer jusqu'au dernier step
        for _ in range(min(100, len(features_df) - 51)):
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, _, done, _, _ = env.step(action)
            if done:
                break

        # Dernière action = signal courant
        final_action, _ = self.model.predict(obs, deterministic=deterministic)
        position = float(np.clip(final_action[0], -1.0, 1.0))
        direction = 1 if position > 0.1 else (-1 if position < -0.1 else 0)
        confidence = min(abs(position), 1.0)

        return {
            "direction": direction,
            "confidence": confidence,
            "position": position,
            "model": f"rl_{self.algo_name.lower()}",
        }

    def load(self, path: Optional[str] = None):
        """Charge un agent pré-entraîné."""
        model_path = path or str(self.checkpoint_dir / "best_model")

        # Détecter le type d'algo sauvegardé
        try:
            self.model = SAC.load(model_path)
            self.algo_name = "SAC"
            logger.info(f"Agent SAC chargé depuis {model_path}")
        except Exception:
            try:
                self.model = PPO.load(model_path)
                self.algo_name = "PPO"
                logger.info(f"Agent PPO chargé depuis {model_path}")
            except Exception as e:
                logger.error(f"Impossible de charger le modèle RL : {e}")
                raise

        vec_norm_path = self.checkpoint_dir / "vec_normalize.pkl"
        if vec_norm_path.exists():
            logger.info("VecNormalize stats chargées")
