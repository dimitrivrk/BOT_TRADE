"""Test connexion Binance Demo Futures."""
import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('config/.env')

api_key = os.getenv('BINANCE_TESTNET_API_KEY', '').strip()
secret  = os.getenv('BINANCE_TESTNET_SECRET_KEY', '').strip()
print(f'Key len: {len(api_key)}, Secret len: {len(secret)}')

try:
    from binance.um_futures import UMFutures
    client = UMFutures(key=api_key, secret=secret, base_url='https://demo-fapi.binance.com')
    balance = client.balance(recvWindow=6000)
    print('Balance OK:')
    for b in balance:
        if float(b.get('balance', 0)) > 0:
            print(f"  {b['asset']}: {b['balance']}")
except Exception as e:
    print(f'Erreur binance-futures-connector: {e}')

    # Fallback: test avec CCXT
    print('\nTest CCXT...')
    from execution.broker import ExchangeBroker
    broker = ExchangeBroker()
    bal = broker.get_balance()
    print(f'Balance CCXT: {bal}')
