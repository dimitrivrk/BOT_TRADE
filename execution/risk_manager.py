"""
Gestionnaire de risque en temps réel.
Contrôle : taille des positions, stop loss, drawdown, circuit breakers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import yaml

from data.processors.indicators import atr
from utils.logger import setup_logger

logger = setup_logger("execution.risk_manager")


class RiskManager:
    """
    Gestion du risque pour le trading live.
    Implémente : Kelly, ATR-based SL/TP, circuit breakers, position limits.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.capital_cfg = cfg["capital"]
        self.risk_cfg = cfg["risk"]

        self.total_capital = self.capital_cfg["total_usdt"]
        self.max_position_pct = self.capital_cfg["max_position_pct"]
        self.max_drawdown_pct = self.capital_cfg["max_drawdown_pct"]
        self.risk_per_trade = self.capital_cfg["risk_per_trade_pct"]

        self.sl_method = self.risk_cfg["stop_loss"]["method"]
        self.sl_atr_mult = self.risk_cfg["stop_loss"].get("atr_multiplier", 2.0)
        self.tp_method = self.risk_cfg["take_profit"]["method"]
        self.rr_ratio = self.risk_cfg["take_profit"].get("rr_ratio", 2.5)

        self.position_sizing = self.risk_cfg.get("position_sizing", "kelly")
        self.kelly_fraction = self.risk_cfg.get("kelly_fraction", 0.25)

        self.max_consecutive_losses = self.risk_cfg.get("max_consecutive_losses", 5)
        self.daily_loss_limit = self.risk_cfg.get("daily_loss_limit_pct", 0.05)

        # État en temps réel
        self.current_capital = self.total_capital
        self.peak_capital = self.total_capital
        self.open_positions: Dict[str, dict] = {}
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.bot_stopped = False
        self.trade_history: List[dict] = []

    # -------------------------------------------------------------------------
    # POSITION SIZING
    # -------------------------------------------------------------------------

    def compute_position_size(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        entry_price: float,
        ohlcv_df: pd.DataFrame,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
    ) -> Dict:
        """
        Calcule la taille optimale de la position.

        Args:
            symbol: Symbole
            direction: 1=long, -1=short
            confidence: Confiance du signal (0-1)
            entry_price: Prix d'entrée anticipé
            ohlcv_df: OHLCV pour le calcul de l'ATR
            win_rate: Taux de réussite historique (pour Kelly)
            avg_win_loss_ratio: Rapport gain/perte moyen (pour Kelly)

        Returns:
            {
                'quantity': float,         # quantité à acheter/vendre
                'usdt_amount': float,      # montant en USDT
                'stop_loss': float,        # prix de stop loss
                'take_profit': float,      # prix de take profit
                'risk_amount': float,      # montant risqué
                'position_pct': float,     # % du capital
            }
        """
        if self.bot_stopped:
            logger.warning("Bot arrêté (circuit breaker). Aucune position.")
            return {}

        # Vérifier les limites de drawdown
        current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_dd >= self.max_drawdown_pct:
            logger.error(f"Drawdown limite atteint ({current_dd:.2%}). Bot arrêté!")
            self.bot_stopped = True
            return {}

        # Vérifier la perte journalière
        daily_loss_pct = self.daily_pnl / self.current_capital
        if daily_loss_pct <= -self.daily_loss_limit:
            logger.warning(f"Limite de perte journalière atteinte ({daily_loss_pct:.2%}). Pause jusqu'à demain.")
            return {}

        # Vérifier les pertes consécutives
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"{self.consecutive_losses} pertes consécutives. Réduction de taille.")
            confidence *= 0.5  # réduire la confiance de moitié

        # Calculer SL
        stop_loss, take_profit = self._compute_sl_tp(
            entry_price, direction, ohlcv_df
        )

        # Distance au SL en %
        sl_distance_pct = abs(entry_price - stop_loss) / entry_price

        # Calcul de la taille selon la méthode
        if self.position_sizing == "kelly" and win_rate and avg_win_loss_ratio:
            position_pct = self._kelly_criterion(win_rate, avg_win_loss_ratio, confidence)
        elif self.position_sizing == "volatility_parity":
            position_pct = self._volatility_parity(ohlcv_df, confidence)
        else:
            # Fixed fractional : risquer risk_per_trade % du capital
            position_pct = (self.risk_per_trade * confidence) / (sl_distance_pct + 1e-8)

        # Appliquer les limites
        position_pct = float(np.clip(position_pct, 0.01, self.max_position_pct))

        # Vérifier le capital disponible
        available_capital = self.current_capital * (1 - self._get_used_capital_pct())
        usdt_amount = min(
            self.current_capital * position_pct,
            available_capital * 0.95,
        )

        if usdt_amount < 10:  # minimum 10 USDT
            logger.warning(f"Capital disponible insuffisant : {usdt_amount:.2f} USDT")
            return {}

        quantity = usdt_amount / entry_price
        risk_amount = quantity * abs(entry_price - stop_loss)

        result = {
            'quantity': float(quantity),
            'usdt_amount': float(usdt_amount),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'risk_amount': float(risk_amount),
            'position_pct': float(position_pct),
            'sl_distance_pct': float(sl_distance_pct),
        }

        logger.info(
            f"Position sizing {symbol} : {usdt_amount:.2f} USDT ({position_pct:.2%}) | "
            f"SL={stop_loss:.4f} (-{sl_distance_pct:.2%}) | TP={take_profit:.4f} | "
            f"Risque={risk_amount:.2f} USDT"
        )

        return result

    def _compute_sl_tp(
        self,
        entry_price: float,
        direction: int,
        ohlcv_df: pd.DataFrame,
    ) -> tuple:
        """Calcule les niveaux SL et TP."""
        if self.sl_method == "atr" and not ohlcv_df.empty:
            atr_val = float(atr(ohlcv_df, 14).iloc[-1])
            sl_distance = atr_val * self.sl_atr_mult
        else:
            sl_distance = entry_price * 0.02  # 2% fixe par défaut

        if direction == 1:  # long
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * self.rr_ratio
        else:  # short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * self.rr_ratio

        return stop_loss, take_profit

    def _kelly_criterion(
        self,
        win_rate: float,
        avg_win_loss_ratio: float,
        confidence: float,
    ) -> float:
        """
        Critère de Kelly fractionné.
        f = (p * b - q) / b, avec b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        p = win_rate * confidence
        q = 1 - p
        b = avg_win_loss_ratio

        kelly_full = (p * b - q) / (b + 1e-8)
        kelly_full = max(0, kelly_full)  # pas de position si négatif

        return kelly_full * self.kelly_fraction

    def _volatility_parity(self, ohlcv_df: pd.DataFrame, confidence: float) -> float:
        """
        Volatility parity : ajuste la taille pour cibler une volatilité constante.
        Taille ∝ 1 / volatilité_récente
        """
        if ohlcv_df.empty or len(ohlcv_df) < 20:
            return 0.05

        log_returns = np.log(ohlcv_df['close'] / ohlcv_df['close'].shift(1)).dropna()
        vol_20 = float(log_returns.tail(20).std() * np.sqrt(252 * 24))  # annualisée

        target_vol = 0.20  # cible 20% de volatilité annualisée
        size = (target_vol / (vol_20 + 1e-8)) * confidence
        return float(np.clip(size, 0.01, self.max_position_pct))

    def _get_used_capital_pct(self) -> float:
        """Retourne le % du capital déjà engagé dans des positions ouvertes."""
        if not self.open_positions:
            return 0.0
        used = sum(p.get('usdt_amount', 0) for p in self.open_positions.values())
        return used / (self.current_capital + 1e-8)

    # -------------------------------------------------------------------------
    # GESTION DES POSITIONS ET PNL
    # -------------------------------------------------------------------------

    def register_trade(self, trade: dict):
        """Enregistre un nouveau trade ouvert."""
        symbol = trade['symbol']
        self.open_positions[symbol] = trade
        logger.info(f"Position ouverte : {symbol} {trade['side']} {trade['quantity']:.6f} @ {trade['price']:.4f}")

    def close_trade(self, symbol: str, close_price: float, reason: str = "signal") -> Optional[dict]:
        """Ferme une position et met à jour les stats."""
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions.pop(symbol)
        entry_price = pos['price']
        quantity = pos['quantity']
        side = pos['side']

        # PnL brut
        if side == 'buy':
            gross_pnl = (close_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - close_price) * quantity

        fee = (close_price * quantity) * 0.001  # 0.1% fee
        net_pnl = gross_pnl - fee

        # Mise à jour du capital
        self.current_capital += net_pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.daily_pnl += net_pnl

        # Streak de pertes
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        trade_result = {
            **pos,
            'close_price': close_price,
            'close_reason': reason,
            'pnl': net_pnl,
            'pnl_pct': net_pnl / (entry_price * quantity),
            'fee': fee,
        }
        self.trade_history.append(trade_result)

        logger.info(
            f"Position fermée : {symbol} | PnL={net_pnl:+.2f} USDT ({trade_result['pnl_pct']:+.2%}) | "
            f"Capital={self.current_capital:.2f} | Raison={reason}"
        )

        return trade_result

    def check_sl_tp(self, symbol: str, current_price: float) -> Optional[str]:
        """Vérifie si le SL ou TP est touché. Retourne la raison ou None."""
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]
        sl = pos.get('stop_loss')
        tp = pos.get('take_profit')
        side = pos.get('side', 'buy')

        if sl and tp:
            if side == 'buy':
                if current_price <= sl:
                    return "stop_loss"
                if current_price >= tp:
                    return "take_profit"
            else:
                if current_price >= sl:
                    return "stop_loss"
                if current_price <= tp:
                    return "take_profit"
        return None

    def reset_daily_pnl(self):
        """Réinitialise le PnL journalier (à appeler à minuit UTC)."""
        logger.info(f"Réinitialisation PnL journalier : {self.daily_pnl:+.2f} USDT")
        self.daily_pnl = 0.0

    def get_status(self) -> dict:
        """Retourne l'état courant du gestionnaire de risque."""
        dd = (self.peak_capital - self.current_capital) / self.peak_capital
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': dd,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'open_positions': len(self.open_positions),
            'used_capital_pct': self._get_used_capital_pct(),
            'bot_stopped': self.bot_stopped,
        }
