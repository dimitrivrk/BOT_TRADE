"""
Gestionnaire de risque en temps reel.
Controle : taille des positions, trailing stop, SL/TP adaptatifs,
take profit partiel, drawdown, circuit breakers.
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
    Implemente : Kelly, trailing stop, SL/TP adaptatifs par regime,
    take profit partiel, circuit breakers, position limits.
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

        # --- Trailing Stop config ---
        trailing_cfg = self.risk_cfg.get("trailing_stop", {})
        self.trailing_enabled = trailing_cfg.get("enabled", True)
        self.trailing_activation_rr = trailing_cfg.get("activation_rr", 1.0)
        self.trailing_atr_mult = trailing_cfg.get("atr_multiplier", 1.5)
        self.trailing_step_pct = trailing_cfg.get("step_pct", 0.002)

        # --- Partial Take Profit config ---
        partial_cfg = self.risk_cfg.get("partial_take_profit", {})
        self.partial_tp_enabled = partial_cfg.get("enabled", True)
        self.partial_tp_levels = partial_cfg.get("levels", [
            {"rr": 1.5, "close_pct": 0.50},   # fermer 50% a 1.5 RR
            {"rr": 2.5, "close_pct": 0.30},   # fermer 30% a 2.5 RR
            # les 20% restants courent avec trailing stop
        ])

        # --- Regime-adaptive SL/TP multipliers ---
        regime_cfg = self.risk_cfg.get("regime_adaptation", {})
        self.regime_adapt_enabled = regime_cfg.get("enabled", True)
        self.regime_multipliers = regime_cfg.get("multipliers", {
            "trending_bull": {"sl_mult": 2.5, "rr_ratio": 3.0},
            "trending_bear": {"sl_mult": 2.5, "rr_ratio": 3.0},
            "ranging":       {"sl_mult": 1.5, "rr_ratio": 2.0},
            "volatile":      {"sl_mult": 3.0, "rr_ratio": 2.0},
        })

        # Etat en temps reel
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
        regime: Optional[str] = None,
        multi_horizon: Optional[Dict] = None,
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

        # Calculer SL (adapte au regime)
        self._last_atr = 0.0
        stop_loss, take_profit = self._compute_sl_tp(
            entry_price, direction, ohlcv_df, regime=regime
        )

        # Ajuster SL/TP avec les predictions multi-horizon de Mamba
        if multi_horizon:
            tp_mult = multi_horizon.get('tp_multiplier', 1.0)
            sl_mult = multi_horizon.get('sl_multiplier', 1.0)
            if direction == 1:
                tp_distance = take_profit - entry_price
                sl_distance_orig = entry_price - stop_loss
                take_profit = entry_price + tp_distance * tp_mult
                stop_loss = entry_price - sl_distance_orig * sl_mult
            else:
                tp_distance = entry_price - take_profit
                sl_distance_orig = stop_loss - entry_price
                take_profit = entry_price - tp_distance * tp_mult
                stop_loss = entry_price + sl_distance_orig * sl_mult
            logger.debug(
                f"Multi-horizon ajustement: SL_mult={sl_mult:.2f}, TP_mult={tp_mult:.2f}, "
                f"duree_estimee={multi_horizon.get('move_duration_hours', '?')}h"
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
            'atr_value': float(self._last_atr),
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
        regime: Optional[str] = None,
    ) -> tuple:
        """
        Calcule les niveaux SL et TP, adaptes au regime de marche.

        Regimes:
          trending_bull/bear: SL large (2.5x ATR), TP genereux (3x RR) pour laisser courir
          ranging:            SL serre (1.5x ATR), TP modeste (2x RR) pour capturer les rebounds
          volatile:           SL large (3x ATR) pour eviter le bruit, TP modeste (2x RR)
        """
        # ATR de base
        if self.sl_method == "atr" and not ohlcv_df.empty:
            atr_val = float(atr(ohlcv_df, 14).iloc[-1])
        else:
            atr_val = entry_price * 0.01  # fallback 1%

        # Multipliers par defaut
        sl_mult = self.sl_atr_mult
        rr_ratio = self.rr_ratio

        # Adapter selon le regime si active
        if self.regime_adapt_enabled and regime and regime in self.regime_multipliers:
            rm = self.regime_multipliers[regime]
            sl_mult = rm.get("sl_mult", sl_mult)
            rr_ratio = rm.get("rr_ratio", rr_ratio)
            logger.debug(f"Regime {regime} : SL mult={sl_mult}, RR={rr_ratio}")

        sl_distance = atr_val * sl_mult

        if direction == 1:  # long
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * rr_ratio
        else:  # short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * rr_ratio

        # Stocker l'ATR dans la position pour le trailing stop
        self._last_atr = atr_val

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
        """Enregistre un nouveau trade ouvert avec tracking trailing stop."""
        symbol = trade['symbol']

        # Ajouter les champs de tracking
        trade['initial_quantity'] = trade['quantity']
        trade['remaining_quantity'] = trade['quantity']
        trade['highest_price'] = trade['price']   # pour trailing long
        trade['lowest_price'] = trade['price']     # pour trailing short
        trade['trailing_active'] = False
        trade['partial_tp_hits'] = []              # niveaux deja touches
        trade['atr_at_entry'] = trade.get('atr_value', 0.0)

        self.open_positions[symbol] = trade
        logger.info(f"Position ouverte : {symbol} {trade['side']} {trade['quantity']:.6f} @ {trade['price']:.4f}")

    def close_trade(self, symbol: str, close_price: float, reason: str = "signal") -> Optional[dict]:
        """
        Ferme une position (totale ou partielle).
        Si reason commence par 'partial_tp_', on ne ferme qu'une fraction.
        """
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]
        entry_price = pos['price']
        side = pos['side']

        # --- Partial close ---
        is_partial = reason.startswith("partial_tp_")
        if is_partial:
            qty_to_close = pos.pop('partial_close_qty', 0)
            if qty_to_close <= 0:
                return None
            # S'assurer qu'on ne ferme pas plus que ce qu'on a
            remaining = pos.get('remaining_quantity', pos['quantity'])
            qty_to_close = min(qty_to_close, remaining)
            quantity = qty_to_close
            pos['remaining_quantity'] = remaining - qty_to_close
            pos['quantity'] = pos['remaining_quantity']
            logger.info(
                f"[PARTIAL CLOSE] {symbol} : ferme {qty_to_close:.6f}, reste {pos['remaining_quantity']:.6f}"
            )
            # Si il reste presque rien, fermer tout
            if pos['remaining_quantity'] < pos['initial_quantity'] * 0.05:
                logger.info(f"[PARTIAL CLOSE] {symbol} : reste < 5%, fermeture totale")
                self.open_positions.pop(symbol)
                is_partial = False
                quantity = qty_to_close + pos['remaining_quantity']
        else:
            # Full close
            pos = self.open_positions.pop(symbol)
            quantity = pos.get('remaining_quantity', pos['quantity'])

        # PnL brut
        if side == 'buy':
            gross_pnl = (close_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - close_price) * quantity

        fee = (close_price * quantity) * 0.001  # 0.1% fee
        net_pnl = gross_pnl - fee

        # Mise a jour du capital
        self.current_capital += net_pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.daily_pnl += net_pnl

        # Streak de pertes (seulement pour les fermetures totales)
        if not is_partial:
            if net_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

        trade_result = {
            **pos,
            'close_price': close_price,
            'close_reason': reason,
            'pnl_usdt': net_pnl,
            'pnl_pct': net_pnl / (entry_price * quantity + 1e-8),
            'fee': fee,
            'quantity_closed': quantity,
            'is_partial': is_partial,
            'entry_price': entry_price,
            'side': side,
        }
        self.trade_history.append(trade_result)

        label = "Partial close" if is_partial else "Position fermee"
        logger.info(
            f"{label} : {symbol} | PnL={net_pnl:+.2f} USDT ({trade_result['pnl_pct']:+.2%}) | "
            f"Capital={self.current_capital:.2f} | Raison={reason}"
        )

        return trade_result

    def check_sl_tp(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Verifie SL/TP avec :
          1. Trailing stop dynamique (suit le prix favorable)
          2. Partial take profit (ferme une fraction aux niveaux intermediaires)
          3. SL classique si trailing pas encore active

        Retourne la raison ou None.
        """
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]
        sl = pos.get('stop_loss')
        tp = pos.get('take_profit')
        side = pos.get('side', 'buy')
        entry = pos.get('price', 0)
        atr_val = pos.get('atr_at_entry', 0)

        if not sl or not tp:
            return None

        is_long = (side == 'buy')

        # --- 1. Mettre a jour le prix extreme (pour trailing) ---
        if is_long:
            pos['highest_price'] = max(pos.get('highest_price', entry), current_price)
        else:
            pos['lowest_price'] = min(pos.get('lowest_price', entry), current_price)

        # --- 2. Verifier activation du trailing stop ---
        if self.trailing_enabled and atr_val > 0:
            sl_distance = abs(entry - sl)
            activation_profit = sl_distance * self.trailing_activation_rr

            if is_long:
                profit = current_price - entry
                if profit >= activation_profit and not pos.get('trailing_active'):
                    pos['trailing_active'] = True
                    logger.info(f"[TRAILING] {symbol} LONG active @ {current_price:.2f} (profit={profit:.2f})")

                if pos.get('trailing_active'):
                    # Trailing SL = plus haut - N x ATR
                    trail_sl = pos['highest_price'] - atr_val * self.trailing_atr_mult
                    # Ne jamais descendre le SL (toujours monter)
                    if trail_sl > sl:
                        old_sl = sl
                        pos['stop_loss'] = trail_sl
                        sl = trail_sl
                        if trail_sl - old_sl >= current_price * self.trailing_step_pct:
                            logger.info(f"[TRAILING] {symbol} SL monte: {old_sl:.2f} -> {trail_sl:.2f}")
            else:
                profit = entry - current_price
                if profit >= activation_profit and not pos.get('trailing_active'):
                    pos['trailing_active'] = True
                    logger.info(f"[TRAILING] {symbol} SHORT active @ {current_price:.2f} (profit={profit:.2f})")

                if pos.get('trailing_active'):
                    trail_sl = pos['lowest_price'] + atr_val * self.trailing_atr_mult
                    if trail_sl < sl:
                        old_sl = sl
                        pos['stop_loss'] = trail_sl
                        sl = trail_sl
                        if old_sl - trail_sl >= current_price * self.trailing_step_pct:
                            logger.info(f"[TRAILING] {symbol} SL baisse: {old_sl:.2f} -> {trail_sl:.2f}")

        # --- 3. Verifier partial take profit ---
        if self.partial_tp_enabled:
            partial_hit = self._check_partial_tp(symbol, current_price)
            if partial_hit:
                return partial_hit

        # --- 4. Verifier SL (trailing ou fixe) ---
        if is_long:
            if current_price <= sl:
                label = "trailing_stop" if pos.get('trailing_active') else "stop_loss"
                return label
        else:
            if current_price >= sl:
                label = "trailing_stop" if pos.get('trailing_active') else "stop_loss"
                return label

        # --- 5. Verifier TP final (pour la quantity restante) ---
        if is_long and current_price >= tp:
            return "take_profit_final"
        if not is_long and current_price <= tp:
            return "take_profit_final"

        return None

    def _check_partial_tp(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Verifie si un niveau de partial take profit est atteint.
        Retourne 'partial_tp_X' si oui, None sinon.
        """
        pos = self.open_positions[symbol]
        entry = pos['price']
        sl = pos.get('stop_loss', entry)
        side = pos.get('side', 'buy')
        is_long = (side == 'buy')
        sl_distance = abs(entry - sl)
        already_hit = pos.get('partial_tp_hits', [])

        for i, level in enumerate(self.partial_tp_levels):
            level_name = f"partial_tp_{i+1}"
            if level_name in already_hit:
                continue  # deja pris

            rr = level['rr']
            target_profit = sl_distance * rr

            if is_long:
                target_price = entry + target_profit
                if current_price >= target_price:
                    close_pct = level['close_pct']
                    qty_to_close = pos['initial_quantity'] * close_pct
                    pos['partial_tp_hits'].append(level_name)
                    pos['partial_close_qty'] = qty_to_close
                    logger.info(
                        f"[PARTIAL TP] {symbol} LONG hit {level_name} @ {current_price:.2f} "
                        f"(target={target_price:.2f}, ferme {close_pct:.0%})"
                    )
                    return level_name
            else:
                target_price = entry - target_profit
                if current_price <= target_price:
                    close_pct = level['close_pct']
                    qty_to_close = pos['initial_quantity'] * close_pct
                    pos['partial_tp_hits'].append(level_name)
                    pos['partial_close_qty'] = qty_to_close
                    logger.info(
                        f"[PARTIAL TP] {symbol} SHORT hit {level_name} @ {current_price:.2f} "
                        f"(target={target_price:.2f}, ferme {close_pct:.0%})"
                    )
                    return level_name

        return None

    def reset_daily_pnl(self):
        """Réinitialise le PnL journalier (à appeler à minuit UTC)."""
        logger.info(f"Réinitialisation PnL journalier : {self.daily_pnl:+.2f} USDT")
        self.daily_pnl = 0.0

    def get_status(self) -> dict:
        """Retourne l'etat courant du gestionnaire de risque."""
        dd = (self.peak_capital - self.current_capital) / self.peak_capital
        # Calculer le win rate et total PnL
        full_closes = [t for t in self.trade_history if not t.get('is_partial', False)]
        total_trades = len(full_closes)
        wins = sum(1 for t in full_closes if t.get('pnl_usdt', t.get('pnl', 0)) > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(t.get('pnl_usdt', t.get('pnl', 0)) for t in self.trade_history)
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': dd,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'open_positions': len(self.open_positions),
            'used_capital_pct': self._get_used_capital_pct(),
            'bot_stopped': self.bot_stopped,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
        }
