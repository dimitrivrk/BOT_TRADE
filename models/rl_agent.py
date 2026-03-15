"""
Agent de Reinforcement Learning v3 pour le trading crypto.

Architecture état de l'art 2025 :
  - Ensemble de 3 agents : SAC + PPO + DDPG
  - Reward risk-aware : Differential Sharpe + pénalité CVaR + drawdown
  - Feature extractor SSM-inspired (State Space Model léger)
  - Vote pondéré dynamique entre les 3 agents
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

from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger("models.rl_agent")


# =============================================================================
# ENVIRONNEMENT DE TRADING v3
# =============================================================================

class CryptoTradingEnv(gym.Env):
    """
    Environnement Gym v3 pour le trading crypto.

    Améliorations v3 :
    - CVaR (Conditional Value at Risk) dans la reward
    - Reward = Profit - Costs - λ × CVaR (risk-aware)
    - Holding bonus pour réduire le churning
    - Observation enrichie avec position + drawdown + volatilité réalisée
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
        self.lookback = config.get("lookback", 20)
        self.mode = mode

        # Config
        self.initial_balance = config.get("initial_balance", 10000)
        self.transaction_cost = config.get("transaction_cost", 0.001)
        self.reward_type = config.get("reward_type", "risk_aware")
        self.cvar_alpha = config.get("cvar_alpha", 0.05)  # 5% tail risk
        self.cvar_lambda = config.get("cvar_lambda", 0.5)  # poids de la pénalité CVaR

        # Differential Sharpe Ratio state (EMA)
        self.eta = 0.01
        self.A_prev = 0.0
        self.B_prev = 0.0

        # Observation augmentée : features + [position, drawdown, volatilité réalisée, time_in_position]
        n_extra = 4
        obs_size = self.lookback * self.n_features + n_extra
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

        # Randomiser le point de départ en mode train
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
        self.time_in_position = 0
        self.prev_action_dir = 0

        # Reset DSR state
        self.A_prev = 0.0
        self.B_prev = 0.0

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.lookback
        obs = self.features[start:self.current_step]
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        flat = obs.flatten().astype(np.float32)

        # Features augmentées
        dd = (self.balance - self.peak_balance) / (self.peak_balance + 1e-8)
        recent_vol = np.std(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.0
        extras = np.array([
            self.position,                              # position actuelle
            np.clip(dd, -1.0, 0.0),                    # drawdown courant
            np.clip(recent_vol * 100, 0.0, 5.0),       # volatilité réalisée
            np.clip(self.time_in_position / 100, 0, 1), # temps dans la position
        ], dtype=np.float32)

        return np.concatenate([flat, extras])

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

        # Temps dans la même direction
        current_dir = 1 if new_position > 0.1 else (-1 if new_position < -0.1 else 0)
        if current_dir == self.prev_action_dir and current_dir != 0:
            self.time_in_position += 1
        else:
            self.time_in_position = 0
        self.prev_action_dir = current_dir

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
        Reward risk-aware v3 :
        R = Profit - Costs - λ_cvar × CVaR - λ_dd × Drawdown + Holding Bonus

        Basé sur le papier "Risk-Aware DRL for Crypto Trading" (2025)
        """
        if len(self.returns_history) < 2:
            return 0.0

        R_t = self.returns_history[-1]
        reward = 0.0

        # === 1. Differential Sharpe Ratio ===
        A_new = self.A_prev + self.eta * (R_t - self.A_prev)
        B_new = self.B_prev + self.eta * (R_t**2 - self.B_prev)

        denom = (B_new - A_new**2) ** 1.5 + 1e-12
        if abs(denom) > 1e-10:
            dsr = (B_new * (R_t - A_new) - 0.5 * A_new * (R_t**2 - B_new)) / denom
        else:
            dsr = R_t

        self.A_prev = A_new
        self.B_prev = B_new

        reward += float(np.clip(dsr * 100, -5.0, 5.0))

        # === 2. CVaR (Conditional Value at Risk) ===
        # Pénalise le tail risk — les pires 5% des returns
        if len(self.returns_history) >= 20:
            returns_arr = np.array(self.returns_history[-100:])  # fenêtre glissante
            var_threshold = np.percentile(returns_arr, self.cvar_alpha * 100)
            tail_returns = returns_arr[returns_arr <= var_threshold]
            if len(tail_returns) > 0:
                cvar = np.mean(tail_returns)
                # Pénalité proportionnelle au tail risk
                reward += self.cvar_lambda * cvar * 50  # cvar est négatif → pénalité

        # === 3. Pénalité de turnover ===
        turnover_penalty = -0.08 * position_change
        reward += turnover_penalty

        # === 4. Pénalité de drawdown progressive ===
        if self.peak_balance > 0:
            dd = (self.balance - self.peak_balance) / self.peak_balance
            if dd < -0.05:
                reward += dd * 3.0
            if dd < -0.15:
                reward -= 2.0

        # === 5. Holding bonus (réduit le churning) ===
        if self.time_in_position > 3 and R_t * self.position > 0:
            # Bonus si on est en position depuis >3 steps et que le trade va dans le bon sens
            reward += 0.1 * min(self.time_in_position / 20, 1.0)

        # === 6. PnL brut (signal de base) ===
        reward += R_t * 10.0

        return float(np.clip(reward, -10.0, 10.0))

    def render(self, mode="human"):
        dd = (self.balance - self.peak_balance) / (self.peak_balance + 1e-8)
        print(
            f"Step {self.current_step} | Balance: {self.balance:.2f} | "
            f"Position: {self.position:.2f} | DD: {dd:.2%}"
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

        # CVaR
        var_5 = np.percentile(returns, 5) if len(returns) > 20 else 0
        cvar_5 = np.mean(returns[returns <= var_5]) if np.any(returns <= var_5) else 0

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(total_return / abs(max_dd + 1e-8)),
            "cvar_5pct": float(cvar_5),
            "total_fees": float(self.total_fees),
        }


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor MLP simple et RAPIDE.
    Flatten l'observation + 3 couches denses. ~5x plus rapide que Conv1D.
    Parfait pour un premier run rapide.
    """

    def __init__(self, observation_space: spaces.Box, lookback: int = 10,
                 n_extra: int = 4, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        n_total = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(n_total, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class SSMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor SSM-inspired (précis) pour capturer les dépendances temporelles.
    Utilise Conv1D + Gating. Plus lent mais meilleur résultat.
    """

    def __init__(self, observation_space: spaces.Box, lookback: int = 20,
                 n_extra: int = 4, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        n_total = observation_space.shape[0]
        self.n_extra = n_extra
        self.lookback = lookback
        self.n_features = (n_total - n_extra) // lookback

        # SSM-like block : Conv1D + Gate + Residual
        self.input_proj = nn.Linear(self.n_features, 128)
        self.norm1 = nn.LayerNorm(128)

        # Multi-scale Conv1D (capture short + medium + long patterns)
        self.conv_short = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv_med = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(128, 64, kernel_size=15, padding=7)

        # Gate mechanism (simplified selective scan)
        self.gate = nn.Sequential(
            nn.Linear(192, 192),
            nn.Sigmoid(),
        )

        self.norm2 = nn.LayerNorm(192)

        # Aggregation
        self.agg = nn.Sequential(
            nn.Linear(192 + n_extra, features_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.GELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Séparer features temporelles et extras
        seq_flat = obs[:, :-self.n_extra]
        extras = obs[:, -self.n_extra:]

        # Reshape en séquence
        x = seq_flat.view(-1, self.lookback, self.n_features)
        x = self.input_proj(x)
        x = self.norm1(x)

        # Multi-scale convolutions (B, T, C) → (B, C, T)
        x_t = x.transpose(1, 2)
        c_short = torch.nn.functional.silu(self.conv_short(x_t))   # (B, 64, T)
        c_med = torch.nn.functional.silu(self.conv_med(x_t))       # (B, 64, T)
        c_long = torch.nn.functional.silu(self.conv_long(x_t))     # (B, 64, T)

        # Concat multi-scale → (B, 192, T)
        multi = torch.cat([c_short, c_med, c_long], dim=1)
        multi = multi.transpose(1, 2)  # (B, T, 192)

        # Gating (selective scan simplifiée)
        gate_vals = self.gate(multi)
        multi = multi * gate_vals

        multi = self.norm2(multi)

        # Dernier token + extras
        last_token = multi[:, -1, :]  # (B, 192)
        combined = torch.cat([last_token, extras], dim=1)  # (B, 192 + 4)

        return self.agg(combined)


# =============================================================================
# AGENT RL v3 — ENSEMBLE SAC + PPO + DDPG
# =============================================================================

class RLTradingAgent:
    """
    Agent RL v3 — Ensemble de 3 algorithmes.

    Architecture :
    - SAC : meilleur en haute volatilité (auto-tuning entropie)
    - PPO : le plus stable et robuste
    - DDPG : le plus rapide, bon pour trends clairs

    Le signal final est un vote pondéré par la performance récente de chaque agent.
    Basé sur FinRL ensemble framework (AI4Finance, Columbia).
    """

    ALGOS = {"SAC": SAC, "PPO": PPO, "DDPG": DDPG}

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["models"]["rl"]
        self.env_cfg = self.cfg.get("env", {})
        self.checkpoint_dir = Path(self.cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Ensemble config
        ensemble_cfg = self.cfg.get("ensemble_agents", ["SAC"])
        self.agent_names = ensemble_cfg if isinstance(ensemble_cfg, list) else [ensemble_cfg]

        self.models = {}  # {"SAC": model, "PPO": model, "DDPG": model}
        self.vec_envs = {}
        self.agent_weights = {name: 1.0 / len(self.agent_names) for name in self.agent_names}

        # Compatibilité avec l'ancienne interface (single model)
        self.model = None
        self.algo_name = self.agent_names[0] if self.agent_names else "SAC"

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
        """Entraîne l'ensemble d'agents RL."""
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

        lookback = self.cfg.get("lookback", 10)
        total_timesteps = self.cfg.get("total_timesteps", 100_000)
        lr = float(self.cfg["learning_rate"])
        eval_freq = self.cfg.get("eval_freq", 20000)
        n_eval_episodes = self.cfg.get("n_eval_episodes", 1)
        fe_type = self.cfg.get("feature_extractor", "mlp")
        n_envs = self.cfg.get("n_envs", 8)  # Nombre d'envs parallèles

        logger.info(f"Config: {total_timesteps:,} steps | lookback={lookback} | "
                     f"lr={lr} | extractor={fe_type} | n_envs={n_envs} | agents={self.agent_names}")

        # Entraîner chaque agent de l'ensemble
        for algo_name in self.agent_names:
            logger.info(f"{'='*60}")
            logger.info(f"Entraînement {algo_name} ({total_timesteps:,} timesteps)")
            logger.info(f"{'='*60}")

            # Envs parallèles (SubprocVecEnv = 1 process par env = N cores utilisés)
            if n_envs > 1:
                env = SubprocVecEnv([self._make_env(train_feat, train_prices) for _ in range(n_envs)])
            else:
                env = DummyVecEnv([self._make_env(train_feat, train_prices)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)

            # Validation env (toujours DummyVecEnv — pas besoin de paralléliser l'eval)
            val_env = DummyVecEnv([self._make_env(val_feat, val_prices, mode="val")])
            val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, training=False)

            # Feature extractor : MLP (rapide) ou SSM (précis)
            n_extra = 4
            if fe_type == "mlp":
                fe_class = MLPFeaturesExtractor
                fe_kwargs = dict(lookback=lookback, n_extra=n_extra, features_dim=128)
            else:
                fe_class = SSMFeaturesExtractor
                fe_kwargs = dict(lookback=lookback, n_extra=n_extra, features_dim=256)

            # Checkpoint dir spécifique à l'algo
            algo_ckpt = self.checkpoint_dir / algo_name.lower()
            algo_ckpt.mkdir(parents=True, exist_ok=True)

            # Callbacks (eval moins fréquente = plus rapide)
            callbacks = [
                EvalCallback(
                    val_env,
                    best_model_save_path=str(algo_ckpt),
                    log_path=str(algo_ckpt / "logs"),
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                ),
                CheckpointCallback(
                    save_freq=50000,
                    save_path=str(algo_ckpt),
                    name_prefix=f"rl_{algo_name.lower()}_ckpt",
                ),
            ]

            # Créer le modèle selon l'algo
            model = self._create_model(algo_name, env, lr, lookback, n_extra, fe_kwargs, fe_class)

            logger.info(f"Entraînement {algo_name} : {total_timesteps:,} timesteps | lr={lr}")

            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=False,
            )

            # Sauvegarder
            env.save(str(algo_ckpt / "vec_normalize.pkl"))
            model.save(str(algo_ckpt / "final_model"))

            self.models[algo_name] = model
            self.vec_envs[algo_name] = env
            logger.info(f"Agent {algo_name} entraîné et sauvegardé !")

        # Garder compatibilité avec l'interface single model
        if self.agent_names:
            self.model = self.models.get(self.agent_names[0])

        logger.info(f"Ensemble entraîné : {list(self.models.keys())}")
        return self.models

    def _create_model(self, algo_name, env, lr, lookback, n_extra, fe_kwargs, fe_class=None):
        """Crée un modèle SB3 selon le nom de l'algorithme."""
        if fe_class is None:
            fe_class = MLPFeaturesExtractor

        if algo_name == "SAC":
            policy_kwargs = dict(
                features_extractor_class=fe_class,
                features_extractor_kwargs=fe_kwargs,
                net_arch=[256, 128],
                activation_fn=nn.ReLU,
            )
            return SAC(
                "MlpPolicy", env,
                learning_rate=lr,
                buffer_size=int(self.cfg.get("buffer_size", 500_000)),
                learning_starts=int(self.cfg.get("learning_starts", 5000)),
                batch_size=512,
                tau=float(self.cfg.get("tau", 0.005)),
                gamma=float(self.cfg.get("gamma", 0.99)),
                train_freq=4,
                gradient_steps=1,
                ent_coef="auto",
                target_entropy="auto",
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/rl",
                device="auto",
            )

        elif algo_name == "PPO":
            policy_kwargs = dict(
                features_extractor_class=fe_class,
                features_extractor_kwargs=fe_kwargs,
                net_arch=dict(pi=[256, 128], vf=[256, 128]),
                activation_fn=nn.ReLU,
            )
            return PPO(
                "MlpPolicy", env,
                learning_rate=lr,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=float(self.cfg.get("gamma", 0.99)),
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                target_kl=0.02,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/rl",
                device="auto",
            )

        elif algo_name == "DDPG":
            # DDPG avec Ornstein-Uhlenbeck noise pour l'exploration
            n_actions = 1
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.2 * np.ones(n_actions),
            )
            policy_kwargs = dict(
                features_extractor_class=fe_class,
                features_extractor_kwargs=fe_kwargs,
                net_arch=[256, 128],
                activation_fn=nn.ReLU,
            )
            return DDPG(
                "MlpPolicy", env,
                learning_rate=lr,
                buffer_size=int(self.cfg.get("buffer_size", 500_000)),
                learning_starts=int(self.cfg.get("learning_starts", 5000)),
                batch_size=512,
                tau=float(self.cfg.get("tau", 0.005)),
                gamma=float(self.cfg.get("gamma", 0.99)),
                train_freq=4,
                gradient_steps=1,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/rl",
                device="auto",
            )

        else:
            raise ValueError(f"Algorithme inconnu : {algo_name}")

    def predict(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Génère un signal de trading via ensemble vote.
        Chaque agent vote, le signal final est la moyenne pondérée.
        """
        # Charger les modèles si nécessaire
        if not self.models:
            self.load()

        lookback = self.cfg.get("lookback", 20)
        n_extra = 4

        # Ajuster features
        if self.models:
            first_model = next(iter(self.models.values()))
            expected_obs_size = first_model.observation_space.shape[0]
            expected_n_features = (expected_obs_size - n_extra) // lookback
            if features_df.shape[1] > expected_n_features:
                features_df = features_df.iloc[:, :expected_n_features]

        # Env temporaire pour l'inférence
        env = CryptoTradingEnv(
            features_df.tail(100 + lookback),
            prices_df.tail(100 + lookback),
            self.env_cfg,
            mode="inference",
        )

        # Collecter les prédictions de chaque agent
        agent_predictions = {}

        for algo_name, model in self.models.items():
            try:
                obs, _ = env.reset()
                # Avancer jusqu'au dernier step
                for _ in range(min(80, len(features_df) - lookback - 10)):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, _, done, _, _ = env.step(action)
                    if done:
                        obs, _ = env.reset()

                # Dernière action
                final_action, _ = model.predict(obs, deterministic=deterministic)
                position = float(np.clip(final_action[0], -1.0, 1.0))
                agent_predictions[algo_name] = position

            except Exception as e:
                logger.warning(f"{algo_name} predict failed : {e}")
                agent_predictions[algo_name] = 0.0

        # Ensemble vote pondéré
        if not agent_predictions:
            return {"direction": 0, "confidence": 0.0, "position": 0.0, "model": "rl_ensemble"}

        weighted_position = sum(
            self.agent_weights.get(name, 1.0 / len(agent_predictions)) * pos
            for name, pos in agent_predictions.items()
        )
        weighted_position /= sum(
            self.agent_weights.get(name, 1.0 / len(agent_predictions))
            for name in agent_predictions
        )

        direction = 1 if weighted_position > 0.1 else (-1 if weighted_position < -0.1 else 0)
        confidence = min(abs(weighted_position), 1.0)

        # Bonus de confiance si les agents sont d'accord
        signs = [1 if p > 0.1 else (-1 if p < -0.1 else 0) for p in agent_predictions.values()]
        agreement = abs(sum(signs)) / max(len(signs), 1)
        confidence = min(confidence * (0.5 + 0.5 * agreement), 1.0)

        return {
            "direction": direction,
            "confidence": confidence,
            "position": float(weighted_position),
            "model": "rl_ensemble",
            "agent_votes": agent_predictions,
        }

    def load(self, path: Optional[str] = None):
        """Charge les agents pré-entraînés."""
        for algo_name in self.agent_names:
            algo_ckpt = self.checkpoint_dir / algo_name.lower()
            model_path = str(algo_ckpt / "best_model")
            final_path = str(algo_ckpt / "final_model")

            AlgoClass = self.ALGOS.get(algo_name)
            if AlgoClass is None:
                continue

            # Essayer best_model d'abord, puis final_model
            for p in [model_path, final_path]:
                try:
                    self.models[algo_name] = AlgoClass.load(p)
                    logger.info(f"Agent {algo_name} chargé depuis {p}")
                    break
                except Exception:
                    continue
            else:
                logger.warning(f"Impossible de charger {algo_name}")

        # Fallback : essayer l'ancien format (single model)
        if not self.models:
            old_path = str(self.checkpoint_dir / "best_model")
            for AlgoClass in [SAC, PPO, DDPG]:
                try:
                    model = AlgoClass.load(old_path)
                    name = AlgoClass.__name__
                    self.models[name] = model
                    logger.info(f"Agent {name} chargé depuis {old_path} (ancien format)")
                    break
                except Exception:
                    continue

        # Compatibilité
        if self.models:
            self.model = next(iter(self.models.values()))

    def update_weights(self, performance: Dict[str, float]):
        """
        Met à jour les poids de l'ensemble basé sur la performance récente.
        Utilisé en live/backtest pour adapter le vote dynamiquement.
        """
        total = sum(max(v, 0.01) for v in performance.values())
        self.agent_weights = {
            name: max(performance.get(name, 0.01), 0.01) / total
            for name in self.agent_names
        }
        logger.info(f"Poids ensemble mis à jour : {self.agent_weights}")
