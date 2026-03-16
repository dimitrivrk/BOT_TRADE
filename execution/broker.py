"""
Connecteur exchange unifié.
- Binance : utilise binance-futures-connector (officiel, testnet supporté)
- Autres  : utilise CCXT
"""

import ccxt
import asyncio
import hashlib
import hmac
import time
import requests
from typing import Dict, List, Optional
import os
import yaml
from dotenv import load_dotenv

from utils.logger import setup_logger

load_dotenv("config/.env")
logger = setup_logger("execution.broker")

# URL testnet Binance Futures
BINANCE_TESTNET_URL  = "https://testnet.binancefuture.com"
BINANCE_MAINNET_URL  = "https://fapi.binance.com"


class ExchangeBroker:
    """
    Interface unifiée vers l'exchange.
    Binance  → binance-futures-connector (UMFutures)
    Autres   → CCXT
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.exc_cfg      = cfg["exchange"]
        self.trading_cfg  = cfg["trading"]
        self.sandbox      = self.exc_cfg.get("sandbox", True)
        self.exchange_name = self.exc_cfg["name"]

        self._binance_client = None   # UMFutures si binance
        self._ccxt_exchange  = None   # CCXT pour les autres
        self._markets_loaded = False

        self._init_exchange()

    # -------------------------------------------------------------------------
    # INITIALISATION
    # -------------------------------------------------------------------------

    def _init_exchange(self):
        name = self.exchange_name.upper()
        if self.sandbox:
            api_key = os.getenv(f"{name}_TESTNET_API_KEY", "").strip()
            secret  = os.getenv(f"{name}_TESTNET_SECRET_KEY", "").strip()
        else:
            api_key = os.getenv(f"{name}_API_KEY", "").strip()
            secret  = os.getenv(f"{name}_SECRET_KEY", "").strip()

        if not api_key:
            logger.warning("API key manquante — mode lecture seule")

        if self.exchange_name == "binance":
            self._init_binance(api_key, secret)
        else:
            self._init_ccxt(api_key, secret)

        logger.info(
            f"Exchange initialisé : {self.exchange_name} | "
            f"Sandbox={self.sandbox} | Type={self.trading_cfg.get('market_type')}"
        )

    def _init_binance(self, api_key: str, secret: str):
        """Initialise le connecteur officiel Binance Futures."""
        from binance.um_futures import UMFutures
        base_url = BINANCE_TESTNET_URL if self.sandbox else BINANCE_MAINNET_URL
        self._binance_client = UMFutures(
            key=api_key,
            secret=secret,
            base_url=base_url,
        )
        # Stocker clés pour appels directs (ex: algo orders)
        self._api_key    = api_key
        self._api_secret = secret
        self._base_url   = base_url
        logger.info(f"Binance UMFutures connecté → {base_url}")

    def _sign_request(self, params: dict) -> dict:
        """Signe les paramètres avec HMAC-SHA256 pour l'API Binance."""
        params['timestamp'] = int(time.time() * 1000)
        query = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self._api_secret.encode('utf-8'),
            query.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params

    def _post_algo_order(self, params: dict) -> Optional[dict]:
        """POST direct sur /fapi/v1/algoOrder (migration Binance déc 2025)."""
        url = f"{self._base_url}/fapi/v1/algoOrder"
        signed = self._sign_request(params)
        headers = {'X-MBX-APIKEY': self._api_key}
        r = requests.post(url, params=signed, headers=headers, timeout=10)
        if r.status_code in (200, 201):
            return r.json()
        raise Exception(f"({r.status_code}, {r.json()})")

    def _init_ccxt(self, api_key: str, secret: str):
        """Initialise CCXT pour les exchanges non-Binance."""
        exchange_class = getattr(ccxt, self.exchange_name)
        market_type = self.trading_cfg.get("market_type", "futures")
        self._ccxt_exchange = exchange_class({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "timeout": self.exc_cfg.get("timeout", 30000),
            "options": {
                "defaultType": market_type,
                "adjustForTimeDifference": True,
            },
        })
        if self.sandbox and hasattr(self._ccxt_exchange, 'set_sandbox_mode'):
            self._ccxt_exchange.set_sandbox_mode(True)

    def _load_markets(self):
        if self._ccxt_exchange and not self._markets_loaded:
            self._ccxt_exchange.load_markets()
            self._markets_loaded = True

    # -------------------------------------------------------------------------
    # BALANCE & POSITIONS
    # -------------------------------------------------------------------------

    def get_balance(self) -> Dict:
        """Retourne la balance USDT du compte."""
        try:
            if self._binance_client:
                balances = self._binance_client.balance(recvWindow=6000)
                usdt = next((b for b in balances if b['asset'] == 'USDT'), {})
                free  = float(usdt.get('availableBalance', usdt.get('balance', 0)))
                total = float(usdt.get('balance', 0))
                return {'free': free, 'total': total, 'used': total - free}
            else:
                bal = self._ccxt_exchange.fetch_balance()
                usdt_free  = bal.get('USDT', {}).get('free', 0)
                usdt_total = bal.get('USDT', {}).get('total', 0)
                return {'free': float(usdt_free), 'total': float(usdt_total),
                        'used': float(usdt_total - usdt_free)}
        except Exception as e:
            logger.error(f"Erreur get_balance : {e}")
            return {'free': 0, 'total': 0, 'used': 0}

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Retourne la position ouverte pour un symbole futures."""
        try:
            sym = symbol.replace('/', '')   # BTC/USDT → BTCUSDT
            if self._binance_client:
                positions = self._binance_client.get_position_risk(
                    symbol=sym, recvWindow=6000
                )
                for pos in positions:
                    amt = float(pos.get('positionAmt', 0))
                    if abs(amt) > 0:
                        side = 'long' if amt > 0 else 'short'
                        return {
                            'symbol': symbol,
                            'side': side,
                            'size': abs(amt),
                            'entry_price': float(pos.get('entryPrice', 0)),
                            'unrealized_pnl': float(pos.get('unRealizedProfit', 0)),
                            'leverage': float(pos.get('leverage', 1)),
                        }
            else:
                positions = self._ccxt_exchange.fetch_positions([symbol])
                for pos in positions:
                    if pos['symbol'] == symbol and abs(pos.get('contracts', 0)) > 0:
                        return {
                            'symbol': symbol,
                            'side': pos['side'],
                            'size': float(pos.get('contracts', 0)),
                            'entry_price': float(pos.get('entryPrice', 0)),
                            'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                            'leverage': float(pos.get('leverage', 1)),
                        }
        except Exception as e:
            logger.error(f"Erreur get_position {symbol} : {e}")
        return None

    def get_all_positions(self) -> List[Dict]:
        """Retourne toutes les positions ouvertes."""
        try:
            if self._binance_client:
                positions = self._binance_client.get_position_risk(recvWindow=6000)
                return [p for p in positions if abs(float(p.get('positionAmt', 0))) > 0]
            else:
                positions = self._ccxt_exchange.fetch_positions()
                return [p for p in positions if abs(p.get('contracts', 0)) > 0]
        except Exception as e:
            logger.error(f"Erreur get_all_positions : {e}")
            return []

    def get_ticker(self, symbol: str) -> Dict:
        """Prix actuel d'un symbole."""
        try:
            if self._binance_client:
                sym = symbol.replace('/', '')
                book = self._binance_client.book_ticker(symbol=sym)
                price = self._binance_client.ticker_price(symbol=sym)
                return {
                    'bid': float(book.get('bidPrice', 0)),
                    'ask': float(book.get('askPrice', 0)),
                    'last': float(price.get('price', 0)),
                    'volume': 0.0,
                }
            else:
                ticker = self._ccxt_exchange.fetch_ticker(symbol)
                return {
                    'bid': float(ticker.get('bid', 0)),
                    'ask': float(ticker.get('ask', 0)),
                    'last': float(ticker.get('last', 0)),
                    'volume': float(ticker.get('baseVolume', 0)),
                }
        except Exception as e:
            logger.error(f"Erreur get_ticker {symbol} : {e}")
            return {}

    # -------------------------------------------------------------------------
    # ORDRES
    # -------------------------------------------------------------------------

    # Précision des quantités par symbole (step size Binance Futures)
    QUANTITY_PRECISION = {
        'BTCUSDT': 3, 'ETHUSDT': 3, 'SOLUSDT': 0, 'BNBUSDT': 2,
    }

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Arrondit la quantité selon la précision du symbole."""
        sym = symbol.replace('/', '')
        decimals = self.QUANTITY_PRECISION.get(sym, 3)
        return round(quantity, decimals)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """Place un ordre market."""
        try:
            quantity = self._round_quantity(symbol, quantity)
            if self._binance_client:
                sym = symbol.replace('/', '')
                params = {
                    'symbol': sym,
                    'side': side.upper(),
                    'type': 'MARKET',
                    'quantity': quantity,
                }
                if reduce_only:
                    params['reduceOnly'] = 'true'
                order = self._binance_client.new_order(**params)
            else:
                params = {}
                if reduce_only:
                    params['reduceOnly'] = True
                order = self._ccxt_exchange.create_market_order(
                    symbol, side, quantity, params=params
                )
            logger.info(f"Ordre market {side.upper()} {symbol} : {quantity:.6f}")
            return order
        except Exception as e:
            logger.error(f"Erreur place_market_order {symbol} : {e}")
            return None

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = True,
    ) -> Optional[Dict]:
        """Place un ordre limit."""
        try:
            if self._binance_client:
                sym = symbol.replace('/', '')
                params = {
                    'symbol': sym,
                    'side': side.upper(),
                    'type': 'LIMIT',
                    'quantity': round(quantity, 6),
                    'price': price,
                    'timeInForce': 'GTX' if post_only else 'GTC',
                }
                if reduce_only:
                    params['reduceOnly'] = 'true'
                order = self._binance_client.new_order(**params)
            else:
                params = {'postOnly': post_only}
                if reduce_only:
                    params['reduceOnly'] = True
                order = self._ccxt_exchange.create_limit_order(
                    symbol, side, quantity, price, params=params
                )
            logger.info(f"Ordre limit {side.upper()} {symbol} : {quantity:.6f} @ {price:.4f}")
            return order
        except Exception as e:
            logger.error(f"Erreur place_limit_order {symbol} : {e}")
            return None

    def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> Optional[Dict]:
        """Place un ordre stop loss via Algo Order API (obligatoire depuis dec 2025)."""
        try:
            sl_side = 'SELL' if side.lower() == 'buy' else 'BUY'
            qty = self._round_quantity(symbol, quantity)
            if self._binance_client:
                sym = symbol.replace('/', '')
                order = self._post_algo_order({
                    'symbol':       sym,
                    'side':         sl_side,
                    'type':         'STOP_MARKET',
                    'algoType':     'CONDITIONAL',
                    'quantity':     qty,
                    'triggerPrice': round(stop_price, 2),
                    'timeInForce':  'GTC',
                })
            else:
                order = self._ccxt_exchange.create_order(
                    symbol, type='STOP_MARKET', side=sl_side.lower(),
                    amount=qty,
                    params={'stopPrice': stop_price, 'reduceOnly': True},
                )
            logger.info(f"Stop Loss placé {symbol} @ {stop_price:.2f}")
            return order
        except Exception as e:
            logger.warning(f"SL serveur echoue ({symbol}), gestion client-side active : {str(e)[:80]}")
            return None

    def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        tp_price: float,
    ) -> Optional[Dict]:
        """Place un ordre take profit via Algo Order API (obligatoire depuis dec 2025)."""
        try:
            tp_side = 'SELL' if side.lower() == 'buy' else 'BUY'
            qty = self._round_quantity(symbol, quantity)
            if self._binance_client:
                sym = symbol.replace('/', '')
                order = self._post_algo_order({
                    'symbol':       sym,
                    'side':         tp_side,
                    'type':         'TAKE_PROFIT_MARKET',
                    'algoType':     'CONDITIONAL',
                    'quantity':     qty,
                    'triggerPrice': round(tp_price, 2),
                    'timeInForce':  'GTC',
                })
            else:
                order = self._ccxt_exchange.create_order(
                    symbol, type='TAKE_PROFIT_MARKET', side=tp_side.lower(),
                    amount=qty,
                    params={'stopPrice': tp_price, 'reduceOnly': True},
                )
            logger.info(f"Take Profit placé {symbol} @ {tp_price:.2f}")
            return order
        except Exception as e:
            logger.warning(f"TP serveur non supporté ({symbol}) → gestion client-side : {str(e)[:60]}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Annule un ordre."""
        try:
            if self._binance_client:
                self._binance_client.cancel_order(
                    symbol=symbol.replace('/', ''), orderId=int(order_id)
                )
            else:
                self._ccxt_exchange.cancel_order(order_id, symbol)
            logger.info(f"Ordre annulé : {order_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur cancel_order {order_id} : {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """Annule tous les ordres ouverts sur un symbole."""
        try:
            if self._binance_client:
                self._binance_client.cancel_open_orders(
                    symbol=symbol.replace('/', ''), recvWindow=6000
                )
            else:
                self._ccxt_exchange.cancel_all_orders(symbol)
            logger.info(f"Tous les ordres annulés : {symbol}")
            return True
        except Exception as e:
            logger.error(f"Erreur cancel_all_orders {symbol} : {e}")
            return False

    def close_position(self, symbol: str) -> Optional[Dict]:
        """Ferme la position avec un ordre market."""
        pos = self.get_position(symbol)
        if not pos:
            logger.warning(f"Aucune position ouverte pour {symbol}")
            return None
        close_side = 'sell' if pos['side'] == 'long' else 'buy'
        return self.place_market_order(symbol, close_side, pos['size'], reduce_only=True)

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Définit le levier pour un symbole futures."""
        try:
            if self._binance_client:
                self._binance_client.change_leverage(
                    symbol=symbol.replace('/', ''),
                    leverage=leverage,
                    recvWindow=6000,
                )
            else:
                self._ccxt_exchange.set_leverage(leverage, symbol)
            logger.info(f"Levier {symbol} : {leverage}x")
            return True
        except Exception as e:
            logger.error(f"Erreur set_leverage {symbol} : {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Retourne les ordres ouverts."""
        try:
            if self._binance_client:
                sym = symbol.replace('/', '') if symbol else None
                kwargs = {'recvWindow': 6000}
                if sym:
                    kwargs['symbol'] = sym
                return self._binance_client.get_open_orders(**kwargs)
            else:
                return self._ccxt_exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Erreur get_open_orders : {e}")
            return []
