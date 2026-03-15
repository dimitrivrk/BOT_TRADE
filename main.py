"""
Point d'entrée principal du bot de trading IA.

Modes disponibles :
  python main.py --mode live          # Trading en temps réel
  python main.py --mode backtest      # Backtest sur données historiques
  python main.py --mode download      # Télécharger les données historiques
  python main.py --mode train         # Entraîner les modèles IA
  python main.py --mode walk_forward  # Walk-forward optimization
"""

import asyncio
import argparse
import signal
import sys
from pathlib import Path

# Ajouter le répertoire courant au PYTHONPATH
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
        result = engine.run(
            symbol=symbol,
            start=args.start,
            end=args.end,
        )
        all_results[symbol] = result

    # Afficher le résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ BACKTEST")
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
        print(f"\nWalk-Forward {symbol} : {len(results)} fenêtres")
        for r in results:
            print(
                f"  [{r.get('window_start')} → {r.get('window_end')}] "
                f"Return={r.get('total_return', 0):.2%} | Sharpe={r.get('sharpe_ratio', 0):.2f}"
            )


def run_download(args):
    """Télécharge les données historiques."""
    from data.collectors.historical import HistoricalDataCollector
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    collector = HistoricalDataCollector()
    pairs = args.pairs or config["trading"]["pairs"]
    collector.fetch_all_pairs(pairs=pairs, days=args.days or 730)
    logger.info("Téléchargement terminé !")


def run_train(args):
    """Entraîne les modèles IA."""
    import yaml
    from data.collectors.historical import HistoricalDataCollector
    from data.processors.features import FeatureEngineer
    from models.tft_model import TFTPredictor
    from models.rl_agent import RLTradingAgent

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    collector = HistoricalDataCollector()
    feature_eng = FeatureEngineer()
    pairs = args.pairs or config["trading"]["pairs"]
    tf = config["trading"]["timeframes"]["primary"]
    tf_higher = config["trading"]["timeframes"]["higher"]

    for symbol in pairs:
        logger.info(f"Entraînement modèles pour {symbol}...")

        # Charger les données
        df = collector.load_data(symbol, tf)
        df_higher = collector.load_data(symbol, tf_higher)

        if len(df) < 500:
            logger.warning(f"Données insuffisantes pour {symbol}, skip.")
            continue

        # Features
        features = feature_eng.compute_all(df, higher_tf_df=df_higher if not df_higher.empty else None)

        # Entraîner TFT
        if config["models"]["tft"]["enabled"] and (not args.model or args.model == "tft"):
            logger.info(f"Entraînement TFT pour {symbol}...")
            tft = TFTPredictor()
            tft.train(features, symbol)

        # Entraîner RL
        if config["models"]["rl"]["enabled"] and (not args.model or args.model == "rl"):
            logger.info(f"Entraînement RL pour {symbol}...")
            rl = RLTradingAgent()
            rl.train(features, df)

    logger.info("Entraînement terminé !")


async def run_live(args):
    """Lance le trading en temps réel."""
    from strategies.ml_strategy import MLTradingStrategy

    strategy = MLTradingStrategy()

    # Gestion propre des signaux d'arrêt
    loop = asyncio.get_event_loop()

    def shutdown(sig, frame):
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        strategy.stop()
        loop.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("=== BOT DE TRADING IA DÉMARRÉ ===")
    logger.info("Ctrl+C pour arrêter proprement")

    await strategy.run()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bot de Trading IA Crypto - TFT + RL + Ensemble"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "download", "train", "walk_forward"],
        default="backtest",
        help="Mode d'exécution",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Paires à trader (ex: BTC/USDT ETH/USDT)",
    )
    parser.add_argument("--start", type=str, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Nombre de jours d'historique")
    parser.add_argument("--model", choices=["tft", "rl", "all"], help="Modèle à entraîner")
    parser.add_argument("--train-months", type=int, default=6, help="Mois d'entraînement (walk-forward)")
    parser.add_argument("--test-months", type=int, default=1, help="Mois de test (walk-forward)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Fichier de configuration")

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
