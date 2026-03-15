"""
Point d'entree principal du bot de trading IA v3.

Architecture 2025 :
  - CryptoMamba (SSM) pour la prediction de prix
  - Ensemble RL (SAC + PPO + DDPG) pour la prise de decision
  - XGBoost feature selection pour les features optimales
  - Reward risk-aware avec CVaR

Modes disponibles :
  python main.py --mode train --pairs BTC/USDT --model rl
  python main.py --mode train --pairs BTC/USDT --model mamba
  python main.py --mode train --pairs BTC/USDT --model all
  python main.py --mode backtest
  python main.py --mode download
  python main.py --mode live
"""

import asyncio
import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger

logger = setup_logger("main")


# =============================================================================
# MODES
# =============================================================================

def run_backtest(args):
    """Lance un backtest complet."""
    from backtesting.engine import BacktestEngine
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    engine = BacktestEngine()
    pairs = args.pairs or config["trading"]["pairs"]

    all_results = {}
    for symbol in pairs:
        logger.info(f"Backtest : {symbol}")
        result = engine.run(symbol=symbol, start=args.start, end=args.end)
        all_results[symbol] = result

    print("\n" + "=" * 70)
    print("RESUME BACKTEST")
    print("=" * 70)
    for symbol, metrics in all_results.items():
        print(f"\n{symbol}:")
        print(f"  Return    : {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe    : {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max DD    : {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate  : {metrics.get('win_rate', 0):.2%}")
        print(f"  N Trades  : {metrics.get('n_trades', 0)}")
    print("=" * 70)

    return all_results


def run_download(args):
    """Telecharge les donnees historiques."""
    from data.collectors.historical import HistoricalDataCollector
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    collector = HistoricalDataCollector()
    pairs = args.pairs or config["trading"]["pairs"]
    collector.fetch_all_pairs(pairs=pairs, days=args.days or 730)
    logger.info("Telechargement termine !")


def run_train(args):
    """Entraine les modeles IA v3."""
    import yaml
    from data.collectors.historical import HistoricalDataCollector
    from data.processors.features import FeatureEngineer

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    collector = HistoricalDataCollector()
    feature_eng = FeatureEngineer()
    pairs = args.pairs or config["trading"]["pairs"]
    tf = config["trading"]["timeframes"]["primary"]
    tf_higher = config["trading"]["timeframes"]["higher"]

    for symbol in pairs:
        logger.info(f"Entrainement modeles pour {symbol}...")

        # Charger les donnees
        df = collector.load_data(symbol, tf)
        df_higher = collector.load_data(symbol, tf_higher)

        if len(df) < 500:
            logger.warning(f"Donnees insuffisantes pour {symbol}, skip.")
            continue

        # Features
        features = feature_eng.compute_all(
            df, higher_tf_df=df_higher if not df_higher.empty else None
        )
        logger.info(f"Features : {features.shape[1]} colonnes, {len(features)} lignes")

        # === Feature Selection (XGBoost) ===
        selected_features = features
        if config["models"].get("feature_selection", {}).get("enabled", False):
            if not args.model or args.model in ("all", "features"):
                try:
                    from models.feature_selector import FeatureSelector
                    selector = FeatureSelector()
                    selected_features = selector.fit_transform(features, df)
                    logger.info(f"Features selectionnees : {selected_features.shape[1]} "
                               f"(sur {features.shape[1]})")
                except Exception as e:
                    logger.warning(f"Feature selection echouee : {e}")
                    selected_features = features

        # === Entrainer CryptoMamba ===
        if config["models"].get("mamba", {}).get("enabled", True):
            if not args.model or args.model in ("mamba", "all"):
                logger.info(f"Entrainement CryptoMamba pour {symbol}...")
                try:
                    from models.crypto_mamba import MambaPredictor
                    mamba = MambaPredictor()
                    mamba.train(selected_features, df, symbol)
                    logger.info(f"CryptoMamba entraine pour {symbol}")
                except Exception as e:
                    logger.error(f"CryptoMamba training failed : {e}")

        # === Entrainer TFT (fallback) ===
        if config["models"].get("tft", {}).get("enabled", False):
            if not args.model or args.model in ("tft", "all"):
                logger.info(f"Entrainement TFT pour {symbol}...")
                try:
                    from models.tft_model import TFTPredictor
                    tft = TFTPredictor()
                    tft.train(selected_features, symbol)
                except Exception as e:
                    logger.error(f"TFT training failed : {e}")

        # === Entrainer RL Ensemble ===
        if config["models"]["rl"]["enabled"]:
            if not args.model or args.model in ("rl", "all"):
                logger.info(f"Entrainement RL Ensemble pour {symbol}...")
                try:
                    from models.rl_agent import RLTradingAgent
                    rl = RLTradingAgent()

                    # Utiliser les features selectionnees si dispo
                    rl_features = selected_features if len(selected_features.columns) <= 40 else features
                    rl.train(rl_features, df)
                except Exception as e:
                    logger.error(f"RL training failed : {e}")
                    import traceback
                    traceback.print_exc()

    logger.info("Entrainement termine !")


def run_walk_forward(args):
    """Lance le walk-forward optimization."""
    from backtesting.engine import BacktestEngine
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    engine = BacktestEngine()
    pairs = args.pairs or config["trading"]["pairs"]

    for symbol in pairs:
        results = engine.run_walk_forward(
            symbol=symbol,
            train_months=args.train_months,
            test_months=args.test_months,
        )
        print(f"\nWalk-Forward {symbol} : {len(results)} fenetres")
        for r in results:
            print(
                f"  [{r.get('window_start')} -> {r.get('window_end')}] "
                f"Return={r.get('total_return', 0):.2%} | Sharpe={r.get('sharpe_ratio', 0):.2f}"
            )


async def run_live(args):
    """Lance le trading en temps reel."""
    from strategies.ml_strategy import MLTradingStrategy

    strategy = MLTradingStrategy()
    loop = asyncio.get_event_loop()

    def shutdown(sig, frame):
        logger.info(f"Signal {sig} recu, arret en cours...")
        strategy.stop()
        loop.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("=== BOT DE TRADING IA v3 DEMARRE ===")
    logger.info("Architecture : CryptoMamba + RL Ensemble (SAC+PPO+DDPG)")
    logger.info("Ctrl+C pour arreter proprement")

    await strategy.run()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bot de Trading IA Crypto v3 — CryptoMamba + RL Ensemble"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "download", "train", "walk_forward"],
        default="backtest",
        help="Mode d'execution",
    )
    parser.add_argument(
        "--pairs", nargs="+",
        help="Paires a trader (ex: BTC/USDT ETH/USDT)",
    )
    parser.add_argument("--start", type=str, help="Date de debut (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Nombre de jours d'historique")
    parser.add_argument(
        "--model",
        choices=["mamba", "tft", "rl", "features", "all"],
        help="Modele a entrainer",
    )
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--config", type=str, default="config/config.yaml")

    args = parser.parse_args()

    logger.info(f"Mode : {args.mode.upper()}")

    if args.mode == "live":
        asyncio.run(run_live(args))
    elif args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "download":
        run_download(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "walk_forward":
        run_walk_forward(args)


if __name__ == "__main__":
    main()
