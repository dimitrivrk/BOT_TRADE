"""
Agent de Reinforcement Learning v8 pour le trading crypto.

Architecture EXACTE FinRL Contest 2024 gagnants (SZU-Fin-621) :
  - PPO-Switch : 2 PPO (aggressive + conservative) + DQN discret
  - MAJORITY VOTING sur les actions (pas vote pondéré continu)
  - Actions discrètes pour TOUS les agents (Short/Hold/Long)
  - Reward = log return portfolio (simple, prouvé)
  - 150k timesteps max (Sharpe stabilise à ~100k)
  - 28 envs parallèles pour PPO (optimisé 31 vCPU)
  - PAS de SAC (instable en trading, entropy collapse)
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

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
        # v6: cvar_alpha/cvar_lambda supprimés (reward simplifiée)

        # v6: Episode max length — des episodes plus courts = plus de diversité
        self.max_episode_length = config.get("max_episode_length", 2000)  # ~83 jours en 1h

        # Observation augmentee : features + portfolio state enrichi
        # 10 features augmentees vs 4 avant = meilleure conscience de l'etat
        n_extra = 10
        self.n_extra = n_extra
        obs_size = self.lookback * self.n_features + n_extra
        # v6: bounds plus larges — VecNormalize s'occupe du scaling
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
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

        # v5: Randomiser le point de départ en mode train
        # Minimum 200 steps d'épisode pour que l'agent ait le temps d'apprendre
        min_episode = 200
        if self.mode == "train" and len(self.prices) > self.lookback + min_episode:
            max_start = len(self.prices) - min_episode
            self.current_step = self.np_random.integers(self.lookback, max_start)
        else:
            self.current_step = self.lookback
        self._episode_start = self.current_step

        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.returns_history = []
        self.position_history = []
        self.portfolio_values = [self.initial_balance]
        self.total_fees = 0.0
        self.peak_balance = self.initial_balance
        self.time_in_position = 0
        self.prev_action_dir = 0
        self.entry_price = 0.0
        self.n_wins = 0
        self.n_losses = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.lookback
        obs = self.features[start:self.current_step]
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        flat = obs.flatten().astype(np.float32)

        # 10 portfolio state features (vs 4 avant)
        dd = (self.balance - self.peak_balance) / (self.peak_balance + 1e-8)
        recent_vol = np.std(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.0

        # Unrealized PnL normalise
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * np.sign(self.position)
        else:
            unrealized_pnl = 0.0

        # Sharpe rolling (20 derniers steps)
        if len(self.returns_history) >= 20:
            recent = np.array(self.returns_history[-20:])
            rolling_sharpe = np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(24)
        else:
            rolling_sharpe = 0.0

        # Win rate recente
        total_trades = self.n_wins + self.n_losses
        win_rate = self.n_wins / (total_trades + 1e-8) if total_trades > 0 else 0.5

        extras = np.array([
            self.position,                                       # 1. position actuelle [-1, 1]
            np.clip(dd, -1.0, 0.0),                              # 2. drawdown courant
            np.clip(recent_vol * 100, 0.0, 5.0),                 # 3. volatilite realisee
            np.clip(self.time_in_position / 50, 0, 1),           # 4. temps dans la position
            np.clip(unrealized_pnl * 10, -5.0, 5.0),             # 5. PnL non realise
            np.clip(rolling_sharpe, -3.0, 3.0),                  # 6. Sharpe rolling
            np.clip(win_rate * 2 - 1, -1.0, 1.0),               # 7. win rate [-1, 1]
            np.clip(self.consecutive_wins / 5, 0, 1),            # 8. streak gagnante
            np.clip(self.consecutive_losses / 5, 0, 1),          # 9. streak perdante
            np.clip((self.balance / self.initial_balance - 1) * 5, -5.0, 5.0),  # 10. total return
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

        # Temps dans la meme direction + tracking entry price
        current_dir = 1 if new_position > 0.1 else (-1 if new_position < -0.1 else 0)
        if current_dir == self.prev_action_dir and current_dir != 0:
            self.time_in_position += 1
        else:
            # Direction change -> fermeture implicite du trade
            if self.prev_action_dir != 0 and current_dir != self.prev_action_dir:
                # Evaluer si le trade ferme etait gagnant ou perdant
                if self.entry_price > 0:
                    pnl_sign = (self.prices[min(self.current_step, len(self.prices)-1)] - self.entry_price) * self.prev_action_dir
                    if pnl_sign > 0:
                        self.n_wins += 1
                        self.consecutive_wins += 1
                        self.consecutive_losses = 0
                    else:
                        self.n_losses += 1
                        self.consecutive_losses += 1
                        self.consecutive_wins = 0
            self.time_in_position = 0
            if current_dir != 0:
                self.entry_price = float(self.prices[min(self.current_step, len(self.prices)-1)])
            else:
                self.entry_price = 0.0
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
        steps_in_episode = self.current_step - self._episode_start
        terminated = (
            self.current_step >= len(self.prices) - 1
            or self.balance < self.initial_balance * 0.5  # stop si perd 50%
        )
        # v5: truncate si episode trop long (force la diversité des conditions de marché)
        truncated = (
            self.mode == "train"
            and steps_in_episode >= self.max_episode_length
        )

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
        Reward v6.0 -- Approche FinRL simplifiée qui marche.

        Basé sur les recherches FinRL, arxiv 2506.04358, et les gagnants
        du FinRL Contest 2025 :
          - Le reward principal = changement de valeur du portfolio (simple, clair)
          - Léger penalty drawdown (protection capital)
          - Léger penalty transaction costs (anti-churning)

        Pourquoi simple > complexe :
          - Le critic peut estimer les Q-values correctement
          - Le reward scale est bien calibré pour SAC
          - Pas de signal contradictoire entre composantes
          - FinRL utilise exactement cette approche et obtient des Sharpe > 2.0
        """
        prev_balance = self.portfolio_values[-1]

        # ===================================================================
        # REWARD PRINCIPAL : log return du portfolio
        # C'est ce que FinRL et les implémentations qui marchent utilisent.
        # Le log return est naturellement bien scaled pour SAC.
        # Multiplié par 1000 pour que le signal soit dans un bon range.
        # ===================================================================
        if prev_balance > 0 and self.balance > 0:
            log_return = np.log(self.balance / prev_balance)
        else:
            log_return = 0.0

        reward = log_return * 100.0  # scale: 1% return = 1.0 reward (×1000 divergeait)

        # ===================================================================
        # PENALTY DRAWDOWN (léger) — protège le capital sans noyer le signal
        # Seulement si drawdown > 10%, pénalité proportionnelle
        # ===================================================================
        current_dd = (self.balance - self.peak_balance) / (self.peak_balance + 1e-8)
        if current_dd < -0.10:
            reward += current_dd * 2.0  # dd=-15% → reward -= 0.30

        # ===================================================================
        # PENALTY TRANSACTION COSTS — anti-churning
        # Pénalise les changements de position (scaled modérément)
        # ===================================================================
        reward -= position_change * 0.1

        return float(np.clip(reward, -5.0, 5.0))

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
# ENVIRONNEMENT DISCRET (pour DQN — approche gagnants FinRL Contest)
# =============================================================================

class CryptoTradingEnvDiscrete(gym.Env):
    """
    Environnement Gym avec actions DISCRÈTES : 0=Short, 1=Hold, 2=Long.
    Utilisé par DQN/Double-DQN — plus stable que continuous pour le trading.
    Les gagnants du FinRL Contest 2024 utilisent cette approche.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, features_df, prices_df, config, mode="train"):
        super().__init__()
        self.features = features_df.values.astype(np.float32)
        self.prices = prices_df['close'].values.astype(np.float32)
        self.returns = np.diff(np.log(self.prices))

        self.n_features = self.features.shape[1]
        self.lookback = config.get("lookback", 20)
        self.mode = mode
        self.initial_balance = config.get("initial_balance", 10000)
        self.transaction_cost = config.get("transaction_cost", 0.001)
        self.max_episode_length = config.get("max_episode_length", 2000)

        n_extra = 5  # position, drawdown, volatilité, unrealized_pnl, total_return
        self.n_extra = n_extra
        obs_size = self.lookback * self.n_features + n_extra

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        # 3 actions discrètes : Short (-1), Hold (0), Long (+1)
        self.action_space = spaces.Discrete(3)
        self.ACTION_MAP = {0: -1.0, 1: 0.0, 2: 1.0}

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        min_episode = 200
        if self.mode == "train" and len(self.prices) > self.lookback + min_episode:
            max_start = len(self.prices) - min_episode
            self.current_step = self.np_random.integers(self.lookback, max_start)
        else:
            self.current_step = self.lookback
        self._episode_start = self.current_step

        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.returns_history = []
        self.portfolio_values = [self.initial_balance]
        self.total_fees = 0.0
        self.peak_balance = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        start = self.current_step - self.lookback
        obs = self.features[start:self.current_step]
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        flat = obs.flatten().astype(np.float32)

        dd = (self.balance - self.peak_balance) / (self.peak_balance + 1e-8)
        recent_vol = np.std(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.0
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        unrealized_pnl = 0.0

        extras = np.array([
            self.position,
            np.clip(dd, -1.0, 0.0),
            np.clip(recent_vol * 100, 0.0, 5.0),
            np.clip(unrealized_pnl * 10, -5.0, 5.0),
            np.clip((self.balance / self.initial_balance - 1) * 5, -5.0, 5.0),
        ], dtype=np.float32)

        return np.concatenate([flat, extras])

    def step(self, action):
        new_position = self.ACTION_MAP[int(action)]

        # Transaction cost
        position_change = abs(new_position - self.position)
        fee = position_change * self.transaction_cost * self.balance
        self.total_fees += fee
        self.balance -= fee

        # Market return
        if self.current_step < len(self.returns):
            market_return = self.returns[self.current_step - 1]
        else:
            market_return = 0.0

        # PnL
        pnl = self.position * market_return * self.balance
        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)
        self.position = new_position

        # Reward = log return du portfolio
        prev_balance = self.portfolio_values[-1]
        if prev_balance > 0 and self.balance > 0:
            reward = np.log(self.balance / prev_balance) * 100.0
        else:
            reward = 0.0

        # Drawdown penalty
        current_dd = (self.balance - self.peak_balance) / (self.peak_balance + 1e-8)
        if current_dd < -0.10:
            reward += current_dd * 2.0

        # Transaction cost penalty
        reward -= position_change * 0.1
        reward = float(np.clip(reward, -5.0, 5.0))

        step_return = pnl / (prev_balance + 1e-8)
        self.returns_history.append(step_return)
        self.portfolio_values.append(self.balance)

        self.current_step += 1
        steps_in_episode = self.current_step - self._episode_start
        terminated = (
            self.current_step >= len(self.prices) - 1
            or self.balance < self.initial_balance * 0.5
        )
        truncated = self.mode == "train" and steps_in_episode >= self.max_episode_length

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        info = {"balance": self.balance, "position": self.position,
                "total_return": (self.balance - self.initial_balance) / self.initial_balance}
        return obs, reward, terminated, truncated, info


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor MLP v5 — plus profond avec LayerNorm et résidual.
    """

    def __init__(self, observation_space: spaces.Box, lookback: int = 10,
                 n_extra: int = 4, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        n_total = observation_space.shape[0]

        self.block1 = nn.Sequential(
            nn.Linear(n_total, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.block1(obs)
        x = self.block2(x)
        return self.block3(x)


class SSMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor SSM-inspired v5 — Multi-scale Conv1D + Temporal Attention + Gating.
    Captures court/moyen/long terme + attention sur les timesteps les plus importants.
    """

    def __init__(self, observation_space: spaces.Box, lookback: int = 20,
                 n_extra: int = 10, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        n_total = observation_space.shape[0]
        self.n_extra = n_extra
        self.lookback = lookback
        self.n_features = (n_total - n_extra) // lookback

        # Input projection + norm
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

        # v5: Temporal attention — apprend quels timesteps sont les plus importants
        self.temporal_attn = nn.Sequential(
            nn.Linear(192, 64),
            nn.Tanh(),
            nn.Linear(64, 1),  # score par timestep
        )

        # v5: Residual block pour les extras (portfolio state est critique)
        self.extras_proj = nn.Sequential(
            nn.Linear(n_extra, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Aggregation finale
        self.agg = nn.Sequential(
            nn.Linear(192 + 64, features_dim),
            nn.LayerNorm(features_dim),
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
        c_short = torch.nn.functional.silu(self.conv_short(x_t))
        c_med = torch.nn.functional.silu(self.conv_med(x_t))
        c_long = torch.nn.functional.silu(self.conv_long(x_t))

        # Concat multi-scale → (B, T, 192)
        multi = torch.cat([c_short, c_med, c_long], dim=1).transpose(1, 2)

        # Gating (selective scan simplifiée)
        gate_vals = self.gate(multi)
        multi = multi * gate_vals
        multi = self.norm2(multi)

        # v5: Temporal attention pooling (au lieu de juste prendre le dernier token)
        # L'attention apprend quels timesteps sont les plus informatifs
        attn_scores = self.temporal_attn(multi)            # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (B, T, 1)
        context = (multi * attn_weights).sum(dim=1)         # (B, 192)

        # v5: Traiter les portfolio state features séparément (pas les mélanger avec le temps)
        extras_out = self.extras_proj(extras)  # (B, 64)

        combined = torch.cat([context, extras_out], dim=1)  # (B, 192 + 64)
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

    ALGOS = {"PPO": PPO, "PPO_AGGRESSIVE": PPO, "PPO_CONSERVATIVE": PPO, "DQN": DQN}

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["models"]["rl"]
        self.env_cfg = self.cfg.get("env", {})
        self.checkpoint_dir = Path(self.cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Ensemble config
        ensemble_cfg = self.cfg.get("ensemble_agents", ["PPO_AGGRESSIVE", "PPO_CONSERVATIVE", "DQN"])
        self.agent_names = ensemble_cfg if isinstance(ensemble_cfg, list) else [ensemble_cfg]

        self.models = {}
        self.agent_weights = {name: 1.0 / len(self.agent_names) for name in self.agent_names}

        # Compatibilité avec l'ancienne interface (single model)
        self.model = None
        self.algo_name = self.agent_names[0] if self.agent_names else "PPO_AGGRESSIVE"

    def _make_env(self, features_df, prices_df, mode="train"):
        """Crée un env discret (Short/Hold/Long) — approche SOTA FinRL Contest."""
        def _init():
            env = CryptoTradingEnvDiscrete(features_df, prices_df, self.env_cfg, mode=mode)
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
        """
        Entraîne l'ensemble d'agents RL avec fenêtres glissantes.

        SOTA approach (FinRL Contest 2024-2025):
        - Entraîne sur la dernière fenêtre de données (dernier 80%)
        - Valide sur les 20% restants (les plus récents)
        - 150k timesteps max par agent (évite l'overfit)
        - Ensemble PPO + DQN + SAC avec vote pondéré
        """
        # Aligner features et prices
        common_idx = features_df.index.intersection(prices_df.index)
        features_df = features_df.loc[common_idx]
        prices_df = prices_df.loc[common_idx]
        logger.info(f"RL données alignées : {len(features_df)} lignes, {features_df.shape[1]} features")

        # Split train/val (80/20) — val = données les plus récentes
        split = int(len(features_df) * 0.8)
        train_feat = features_df.iloc[:split]
        train_prices = prices_df.iloc[:split]
        val_feat = val_features_df if val_features_df is not None else features_df.iloc[split:]
        val_prices = val_prices_df if val_prices_df is not None else prices_df.iloc[split:]

        logger.info(f"Train: {len(train_feat)} lignes ({train_feat.index.min()} → {train_feat.index.max()})")
        logger.info(f"Val:   {len(val_feat)} lignes ({val_feat.index.min()} → {val_feat.index.max()})")

        lookback = self.cfg.get("lookback", 10)
        total_timesteps = self.cfg.get("total_timesteps", 100_000)
        lr = float(self.cfg["learning_rate"])
        eval_freq = self.cfg.get("eval_freq", 20000)
        n_eval_episodes = self.cfg.get("n_eval_episodes", 1)
        fe_type = self.cfg.get("feature_extractor", "mlp")
        n_envs = self.cfg.get("n_envs", 8)

        logger.info(f"Config: {total_timesteps:,} steps | lookback={lookback} | "
                     f"lr={lr} | extractor={fe_type} | n_envs={n_envs} | agents={self.agent_names}")

        # Entraîner chaque agent de l'ensemble
        for algo_name in self.agent_names:
            logger.info(f"{'='*60}")
            logger.info(f"Entraînement {algo_name} ({total_timesteps:,} timesteps)")
            logger.info(f"{'='*60}")

            # v8 SOTA: Tous les agents en discret (Short/Hold/Long)
            # PPO: 28 envs parallèles (on-policy, bénéficie du parallélisme)
            # DQN: 1 env (off-policy, replay buffer, pas besoin de parallel)
            is_on_policy = algo_name.startswith("PPO")

            if is_on_policy and n_envs > 1:
                env = SubprocVecEnv([self._make_env(train_feat, train_prices) for _ in range(n_envs)])
            else:
                env = DummyVecEnv([self._make_env(train_feat, train_prices)])
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

            # Validation env (toujours 1 env)
            val_env = DummyVecEnv([self._make_env(val_feat, val_prices, mode="val")])
            val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, training=False)

            # Feature extractor : MLP (rapide) ou SSM (précis)
            n_extra = 5  # v8: 5 portfolio state features (env discrète)
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

            # Sauvegarder sur disque
            env.save(str(algo_ckpt / "vec_normalize.pkl"))
            model.save(str(algo_ckpt / "final_model"))
            logger.info(f"Agent {algo_name} entraîné et sauvegardé !")

            # Fermer et supprimer les envs pour libérer les process/locks
            try:
                val_env.close()
                env.close()
            except Exception:
                pass
            # Libérer toute la mémoire (replay buffer, thread locks, etc.)
            # Indispensable pour que SubprocVecEnv du prochain agent ne crash pas
            del env, val_env, model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Recharger tous les modèles depuis le disque (évite les thread locks en mémoire)
        algo_classes = {"PPO_AGGRESSIVE": PPO, "PPO_CONSERVATIVE": PPO, "DQN": DQN}
        for algo_name in self.agent_names:
            algo_ckpt = self.checkpoint_dir / algo_name.lower()
            model_path = algo_ckpt / "final_model.zip"
            if model_path.exists():
                self.models[algo_name] = algo_classes[algo_name].load(str(algo_ckpt / "final_model"))
                logger.info(f"Modèle {algo_name} rechargé depuis disque")

        # Garder compatibilité avec l'interface single model
        if self.agent_names:
            self.model = self.models.get(self.agent_names[0])

        logger.info(f"Ensemble entraîné : {list(self.models.keys())}")
        return self.models

    def _create_model(self, algo_name, env, lr, lookback, n_extra, fe_kwargs, fe_class=None):
        """
        Crée un modèle SB3 — approche SOTA FinRL Contest 2024.
        3 agents en actions discrètes (Short/Hold/Long) :
          - PPO_AGGRESSIVE : lr=3e-4, ent=0.02, explore plus
          - PPO_CONSERVATIVE : lr=5e-5, ent=0.005, exploite plus
          - DQN : epsilon-greedy, vanilla DQN
        """
        if fe_class is None:
            fe_class = MLPFeaturesExtractor

        if algo_name == "PPO_AGGRESSIVE":
            # PPO agressif : explore plus, apprend vite
            policy_kwargs = dict(
                features_extractor_class=fe_class,
                features_extractor_kwargs=fe_kwargs,
                net_arch=dict(pi=[256, 128], vf=[256, 128]),
                activation_fn=nn.ReLU,
            )
            return PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,              # Agressif : lr haut
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.02,                   # Plus d'exploration
                target_kl=0.03,                  # Plus permissif
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/rl",
                device="cpu",                    # PPO est plus rapide sur CPU (SB3 warning)
            )

        elif algo_name == "PPO_CONSERVATIVE":
            # PPO conservatif : exploite plus, plus stable
            policy_kwargs = dict(
                features_extractor_class=fe_class,
                features_extractor_kwargs=fe_kwargs,
                net_arch=dict(pi=[128, 64], vf=[128, 64]),   # Réseau plus petit
                activation_fn=nn.ReLU,
            )
            return PPO(
                "MlpPolicy", env,
                learning_rate=5e-5,              # Conservatif : lr bas
                n_steps=4096,                    # Plus de data avant update
                batch_size=512,
                n_epochs=5,                      # Moins d'epochs (moins d'overfit)
                gamma=0.99,
                gae_lambda=0.98,                 # GAE plus haut = bias vers long terme
                clip_range=0.15,                 # Clip plus serré
                ent_coef=0.005,                  # Moins d'exploration
                target_kl=0.01,                  # Plus strict
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/rl",
                device="cpu",
            )

        elif algo_name == "DQN":
            # DQN — actions discrètes, epsilon-greedy
            # Approche gagnants FinRL Contest 2024 pour le crypto
            policy_kwargs = dict(
                features_extractor_class=fe_class,
                features_extractor_kwargs=fe_kwargs,
                net_arch=[256, 128],
                activation_fn=nn.ReLU,
            )
            return DQN(
                "MlpPolicy", env,
                learning_rate=1e-4,
                buffer_size=int(self.cfg.get("buffer_size", 200_000)),
                learning_starts=int(self.cfg.get("learning_starts", 2000)),
                batch_size=256,
                tau=float(self.cfg.get("tau", 0.005)),
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.3,         # 30% du training en exploration
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/rl",
                device="auto",                    # DQN bénéficie du GPU
            )

        else:
            raise ValueError(f"Algorithme inconnu : {algo_name}")

    def _load_vec_normalizers(self, features_df, prices_df):
        """Charge et cache les VecNormalize pour chaque agent (appelé une seule fois)."""
        if hasattr(self, '_vec_norms') and self._vec_norms is not None:
            return
        self._vec_norms = {}
        lookback = self.cfg.get("lookback", 20)
        for algo_name in self.models:
            vec_norm_path = self.checkpoint_dir / algo_name.lower() / "vec_normalize.pkl"
            if vec_norm_path.exists():
                try:
                    # v8: Tous les agents utilisent l'env discrète
                    dummy_env = DummyVecEnv([
                        lambda: CryptoTradingEnvDiscrete(
                            features_df.tail(lookback + 10),
                            prices_df.tail(lookback + 10),
                            self.env_cfg, mode="inference"
                        )
                    ])
                    vec_norm = VecNormalize.load(str(vec_norm_path), dummy_env)
                    vec_norm.training = False
                    vec_norm.norm_reward = False
                    self._vec_norms[algo_name] = vec_norm
                    logger.info(f"VecNormalize chargé pour {algo_name}")
                except Exception as e:
                    logger.warning(f"VecNormalize load failed pour {algo_name}: {e}")

    def predict(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Génère un signal de trading via ensemble vote.

        Fix v3.4 :
        - Charge VecNormalize une seule fois (caché)
        - Construit l'observation directement (pas de simulation 80 steps)
        - Fix extraction scalaire depuis action array
        """
        # Charger les modèles si nécessaire
        if not self.models:
            self.load()

        lookback = self.cfg.get("lookback", 48)
        n_extra = 5  # v8: 5 portfolio state features (env discrète)

        # Charger VecNormalize une seule fois
        self._load_vec_normalizers(features_df, prices_df)

        # Ajuster le nombre de features pour matcher le modele
        if self.models:
            first_model = next(iter(self.models.values()))
            expected_obs_size = first_model.observation_space.shape[0]
            expected_n_features = (expected_obs_size - n_extra) // lookback
            if features_df.shape[1] > expected_n_features:
                features_df = features_df.iloc[:, :expected_n_features]

        # --- Construire l'observation directement ---
        if len(features_df) < lookback:
            return {"direction": 0, "confidence": 0.0, "position": 0.0, "model": "rl_ensemble"}

        feat_window = features_df.iloc[-lookback:].values.astype(np.float32)
        feat_window = np.nan_to_num(feat_window, nan=0.0, posinf=5.0, neginf=-5.0)
        flat = feat_window.flatten()

        # Volatilité récente pour les extras
        if len(prices_df) >= 20:
            recent_returns = np.diff(np.log(prices_df['close'].values[-21:].astype(np.float64)))
            recent_vol = float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0
        else:
            recent_vol = 0.0

        # Collecter les prédictions de chaque agent
        agent_predictions = {}

        # v8: Tous les agents sont en discret — obs identique (5 extras)
        extras_discrete = np.array([
            0.0,                                        # position neutre
            0.0,                                        # pas de drawdown
            np.clip(recent_vol * 100, 0.0, 5.0),       # volatilité
            0.0,                                        # unrealized_pnl
            0.0,                                        # total_return
        ], dtype=np.float32)
        obs_discrete = np.concatenate([flat, extras_discrete]).astype(np.float32)

        for algo_name, model in self.models.items():
            try:
                obs = obs_discrete.copy()

                # Normaliser via VecNormalize caché
                if hasattr(self, '_vec_norms') and algo_name in self._vec_norms:
                    obs = self._vec_norms[algo_name].normalize_obs(obs)

                # Prédire — tous les agents retournent une action discrète (0/1/2)
                obs_2d = obs.reshape(1, -1)
                action, _ = model.predict(obs_2d, deterministic=deterministic)

                # Convertir action discrète → direction (-1, 0, +1)
                action_map = {0: -1, 1: 0, 2: 1}
                agent_predictions[algo_name] = action_map.get(int(action.flatten()[0]), 0)

            except Exception as e:
                logger.warning(f"{algo_name} predict failed : {e}")
                agent_predictions[algo_name] = 0

        # ================================================================
        # MAJORITY VOTING — approche exacte des gagnants FinRL Contest 2024
        # Chaque agent vote Short(-1), Hold(0) ou Long(+1)
        # L'action majoritaire gagne
        # ================================================================
        if not agent_predictions:
            return {"direction": 0, "confidence": 0.0, "position": 0.0, "model": "rl_ensemble"}

        votes = list(agent_predictions.values())
        # Compter les votes
        vote_counts = {-1: votes.count(-1), 0: votes.count(0), 1: votes.count(1)}
        # Direction = vote majoritaire
        direction = max(vote_counts, key=vote_counts.get)
        # Confiance = proportion d'accord (3/3 = 1.0, 2/3 = 0.67, 1/3 = 0.33)
        confidence = vote_counts[direction] / max(len(votes), 1)

        return {
            "direction": direction,
            "confidence": confidence,
            "position": float(direction),
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
