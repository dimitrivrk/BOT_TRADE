"""
Test complet : BUY + SL + TP + fermeture + Discord.
Lance avec : python test_trade.py
"""
import asyncio
import yaml
import ccxt
from execution.broker import ExchangeBroker
from utils.discord_notifier import DiscordNotifier

async def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    broker   = ExchangeBroker("config/config.yaml")
    webhook  = cfg.get("discord", {}).get("webhook_url", "")
    discord  = DiscordNotifier(webhook) if webhook.startswith("https://discord") else None
    exchange = ccxt.binance({"options": {"defaultType": "future"}})

    print("\n========================================")
    print("   TEST COMPLET TRADE TESTNET BTC/USDT  ")
    print("========================================")

    # --- Balance ---
    bal     = broker.get_balance()
    balance = bal.get('free', 0)
    print(f"\n[1] Balance       : {balance:.2f} USDT")

    # --- Prix ---
    ticker = exchange.fetch_ticker("BTC/USDT")
    price  = ticker['last']
    print(f"[2] Prix BTC      : ${price:,.2f}")

    # --- Params ordre ---
    quantity  = 0.002
    sl_price  = round(price * 0.98, 1)   # SL -2%
    tp_price  = round(price * 1.04, 1)   # TP +4%
    usdt_val  = round(quantity * price, 2)
    print(f"[3] Ordre prévu   : BUY {quantity} BTC (~{usdt_val} USDT)")
    print(f"    SL = ${sl_price:,.1f}  |  TP = ${tp_price:,.1f}")

    input("\n>>> Appuie sur Entrée pour envoyer l'ordre BUY...")

    # --- Market order ---
    print("\n[4] Envoi ordre market BUY...")
    order = broker.place_market_order("BTC/USDT", "buy", quantity)
    if not order:
        print("    ❌ ECHEC ordre market")
        return
    print("    ✅ Ordre market OK")

    # --- Stop Loss ---
    print("[5] Envoi Stop Loss...")
    sl = broker.place_stop_loss("BTC/USDT", "buy", quantity, sl_price)
    if sl:
        print(f"    ✅ Stop Loss OK @ ${sl_price:,.1f}")
    else:
        print("    ❌ ECHEC Stop Loss")

    # --- Take Profit ---
    print("[6] Envoi Take Profit...")
    tp = broker.place_take_profit("BTC/USDT", "buy", quantity, tp_price)
    if tp:
        print(f"    ✅ Take Profit OK @ ${tp_price:,.1f}")
    else:
        print("    ❌ ECHEC Take Profit")

    # --- Résumé ---
    print("\n--- Résumé ---")
    print(f"    Market order : {'✅' if order else '❌'}")
    print(f"    Stop Loss    : {'✅' if sl else '❌'}")
    print(f"    Take Profit  : {'✅' if tp else '❌'}")

    if discord:
        discord.notify_trade_entry(
            symbol="BTC/USDT", side="buy",
            entry_price=price, quantity=quantity,
            usdt_amount=usdt_val, stop_loss=sl_price,
            take_profit=tp_price, confidence=1.0,
            reason="TEST MANUEL",
        )
        print("    Discord      : ✅ notif envoyée")

    input("\n>>> Appuie sur Entrée pour FERMER la position (annule SL/TP)...")

    # --- Fermeture ---
    print("\n[7] Annulation ordres SL/TP...")
    broker.cancel_all_orders("BTC/USDT")
    print("    ✅ Ordres annulés")

    print("[8] Fermeture position...")
    close = broker.close_position("BTC/USDT")
    if close:
        ticker2    = exchange.fetch_ticker("BTC/USDT")
        exit_price = ticker2['last']
        pnl        = (exit_price - price) * quantity
        print(f"    ✅ Position fermée @ ${exit_price:,.2f}")
        print(f"    P&L estimé : {pnl:+.4f} USDT")
        if discord:
            discord.notify_trade_close(
                symbol="BTC/USDT", side="buy",
                entry_price=price, close_price=exit_price,
                pnl_usdt=pnl, pnl_pct=(exit_price - price) / price * 100,
                reason="TEST MANUEL",
            )
    else:
        print("    ❌ ECHEC fermeture")

    await asyncio.sleep(3)
    print("\n========================================")
    print("              FIN DU TEST               ")
    print("========================================\n")

asyncio.run(main())
