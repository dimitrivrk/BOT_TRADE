"""
Test rapide : simule une nouvelle bougie et verifie signal + Discord.
Lance avec : python test_signal.py
"""
import asyncio
from strategies.ml_strategy import MLTradingStrategy

async def main():
    strat = MLTradingStrategy()
    await strat.initialize()
    print("\n=== Simulation nouvelle bougie BTC/USDT/1h ===")
    strat.on_new_candle("BTC/USDT", "1h")
    print("=== Termine - verifie ton Discord ===")
    await asyncio.sleep(3)  # laisser le thread Discord envoyer

asyncio.run(main())
