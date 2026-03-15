"""
Script de backtest rapide.
Usage: python run_backtest.py [--symbol BTC/USDT] [--start 2024-07-01] [--end 2024-12-31]
"""
import sys
sys.path.insert(0, '.')

import argparse
from backtesting.engine import BacktestEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTC/USDT')
    parser.add_argument('--start', default=None, help='Date début (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='Date fin (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default=None)
    args = parser.parse_args()

    engine = BacktestEngine()
    result = engine.run(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
    )

    if result:
        print("\n" + "=" * 60)
        print("RÉSULTATS BACKTEST")
        print("=" * 60)
        for k, v in result.items():
            if isinstance(v, float):
                if 'return' in k or 'drawdown' in k or 'rate' in k or 'pct' in k:
                    print(f"  {k:25s} : {v:.2%}")
                else:
                    print(f"  {k:25s} : {v:.4f}")
            else:
                print(f"  {k:25s} : {v}")

if __name__ == '__main__':
    main()
