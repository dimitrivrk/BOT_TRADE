"""
Calcul des métriques de performance de trading.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Optional
import logging

logger = logging.getLogger("backtesting.metrics")


class PerformanceMetrics:
    """Calcule l'ensemble des métriques de performance d'un backtest."""

    def compute(self, portfolio: vbt.Portfolio, initial_capital: float = 10000.0) -> Dict:
        """Calcule toutes les métriques depuis un Portfolio vectorbt."""
        try:
            returns = portfolio.returns()
            total_return = float(portfolio.total_return())
            max_dd = float(portfolio.max_drawdown())

            # Sharpe annualisé (vectorbt peut retourner NaN si pas assez de données)
            try:
                sharpe = float(portfolio.sharpe_ratio())
                if np.isnan(sharpe):
                    sharpe = 0.0
            except Exception:
                sharpe = 0.0

            # Nombre de trades — compatible toutes versions vectorbt
            try:
                n_trades_raw = portfolio.trades.count()
                n_trades = int(n_trades_raw) if np.isscalar(n_trades_raw) else int(n_trades_raw.iloc[0])
            except Exception:
                n_trades = 0

            # Win rate
            try:
                win_rate = float(portfolio.trades.win_rate()) if n_trades > 0 else 0.0
                if np.isnan(win_rate):
                    win_rate = 0.0
            except Exception:
                win_rate = 0.0

            # Sortino manuel (plus robuste)
            ann_factor = np.sqrt(252 * 24)
            neg_returns = returns[returns < 0]
            downside_std = float(neg_returns.std()) if len(neg_returns) > 0 else 1e-8
            sortino = float(returns.mean() / (downside_std + 1e-8) * ann_factor)

            # Calmar
            calmar = float(total_return / (abs(max_dd) + 1e-8))

            # Profit Factor depuis les records de trades
            profit_factor = expectancy = avg_win = avg_loss = 0.0
            try:
                trades_df = portfolio.trades.records_readable
                if len(trades_df) > 0:
                    # Chercher la colonne PnL (nom variable selon version vectorbt)
                    pnl_col = next((c for c in trades_df.columns if 'pnl' in c.lower() or 'p&l' in c.lower()), None)
                    if pnl_col:
                        pnl = trades_df[pnl_col]
                        gross_profit = pnl[pnl > 0].sum()
                        gross_loss = abs(pnl[pnl < 0].sum())
                        profit_factor = float(gross_profit / (gross_loss + 1e-8))
                        expectancy = float(pnl.mean())
                        avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
                        avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
            except Exception as e:
                logger.warning(f"Calcul profit factor échoué : {e}")

            logger.debug(
                f"Métriques calculées : n_trades={n_trades}, return={total_return:.2%}, "
                f"sharpe={sharpe:.2f}, max_dd={max_dd:.2%}"
            )

            # final_value et total_fees : noms variables selon version vectorbt
            try:
                final_value = float(portfolio.final_value())
            except Exception:
                final_value = float(portfolio.value().iloc[-1])

            try:
                total_fees = float(portfolio.total_fees())
            except AttributeError:
                try:
                    total_fees = float(portfolio.fees.sum())
                except Exception:
                    total_fees = 0.0

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino if not np.isnan(sortino) else 0.0,
                'calmar_ratio': calmar if not np.isnan(calmar) else 0.0,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'n_trades': n_trades,
                'final_value': final_value,
                'total_fees': total_fees,
            }

        except Exception as e:
            # Logger l'erreur complète pour debug
            logger.error(f"Erreur calcul métriques : {e}", exc_info=True)
            return {
                'error': str(e), 'total_return': 0.0, 'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'n_trades': 0,
            }

    def compute_from_returns(self, returns: pd.Series, freq_hours: int = 1) -> Dict:
        """Calcule les métriques depuis une série de retours."""
        ann_factor = np.sqrt(252 * 24 / freq_hours)

        total_return = float((1 + returns).prod() - 1)
        sharpe = float(returns.mean() / (returns.std() + 1e-8) * ann_factor)

        neg = returns[returns < 0]
        sortino = float(returns.mean() / (neg.std() + 1e-8) * ann_factor) if len(neg) > 0 else 0.0

        # Drawdown
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        dd = (equity - rolling_max) / rolling_max
        max_dd = float(dd.min())

        calmar = float(total_return / (abs(max_dd) + 1e-8))

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'ann_return': float((1 + returns.mean()) ** (252 * 24 / freq_hours) - 1),
            'ann_volatility': float(returns.std() * ann_factor),
        }
