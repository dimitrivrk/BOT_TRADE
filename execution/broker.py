"""
Connecteur exchange via CCXT.
Gère : ordres, positions, balance, historique.
"""

import ccxt
import ccxt.pro as ccxtpro
import asyncio
from typing import Dict, List, Optional
import os
import yaml
from dotenv import load_dotenv

from utils.logger import setup_logger

load_dotenv("config/.env")
logger = setup_logger("execution.broker")


class ExchangeBroker:
    """
    Interface unifiée vers l'exchange via CCXT.
    Gère l'authentification, les ordres et le monitoring des positions.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.exc_cfg = cfg["exchange"]
        self.trading_cfg = cfg["trading"]
        self.sandbox = self.exc_cfg.get("sandbox", True)
        self.exchange_name = self.exc_cfg["name"]

        self.exchange = self._init_exchange()
        self._markets_loaded = False

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialise CCXT avec authentification."""
        name = self.exchange_name.upper()

        if self.sandbox:
            api_key = os.getenv(f"{name}_TESTNET_API_KEY", "")
            secret = os.getenv(f"{name}_TESTNET_SECRET_KEY", "")
        else:
            api_key = os.getenv(f"{name}_API_KEY", "")
            secret = os.getenv(f"{name}_SECRET_KEY", "")

        exchange_class = getattr(ccxt, self.exchange_name)
        exchange = exchange_class({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "timeout": self.exc_cfg.get("timeout", 30000),
            "options": {
                "defaultType": self.trading_cfg.get("market_type", "futures"),
                "adjustForTimeDifference": True,
            },
        })

        if self.sandbox and hasattr(exchange, 'set_sandbox_mode'):
            exchange.set_sandbox_mode(True)

        if not api_key:
            logger.warning("API key manquante - mode lecture seule uniquement")

        logger.info(
            f"Exchange initialisé : {self.exchange_name} | "
            f"Sandbox={self.sandbox} | Type={self.trading_cfg.get('market_type')}"
        )
        return exchange

    def _load_markets(self):
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    # -------------------------------------------------------------------------
    # BALANCE & POSITIONS
    # -------------------------------------------------------------------------

    def get_balance(self) -> Dict:
        """Retourne la balance du compte."""
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = balance.get('USDT', {}).get('free', 0)
            usdt_total = balance.get('USDT', {}).get('total', 0)
            return {
                'free': float(usdt_free),
                'total': float(usdt_total),
                'used': float(usdt_total - usdt_free),
            }
        except Exception as e:
            logger.error(f"Erreur get_balance : {e}")
            return {'free': 0, 'total': 0, 'used': 0}

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Retourne la position ouverte pour un symbole (futures)."""
        try:
            positions = self.exchange.fetch_positions([symbol])
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
            positions = self.exchange.fetch_positions()
            return [
                p for p in positions
                if abs(p.get('contracts', 0)) > 0
            ]
        except Exception as e:
            logger.error(f"Erreur get_all_positions : {e}")
            return []

    def get_ticker(self, symbol: str) -> Dict:
        """Prix actuel d'un symbole."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
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

    def place_market_order(
        self,
        symbol: str,
        side: str,              # 'buy' ou 'sell'
        quantity: float,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """Place un ordre market."""
        try:
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            if self.trading_cfg.get("market_type") == "futures":
                params['positionSide'] = 'LONG' if side == 'buy' else 'SHORT'

            order = self.exchange.create_market_order(
                symbol, side, quantity, params=params
            )
            logger.info(
                f"Ordre market {side.upper()} {symbol} : "
                f"{quantity:.6f} @ market | ID={order.get('id')}"
            )
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
        post_only: bool = True,     # maker order pour fees réduites
    ) -> Optional[Dict]:
        """Place un ordre limit."""
        try:
            params = {'postOnly': post_only}
            if reduce_only:
                params['reduceOnly'] = True

            order = self.exchange.create_limit_order(
                symbol, side, quantity, price, params=params
            )
            logger.info(
                f"Ordre limit {side.upper()} {symbol} : "
                f"{quantity:.6f} @ {price:.4f} | ID={order.get('id')}"
            )
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
        """Place un ordre stop loss."""
        try:
            # Le côté du SL est inverse à la position
            sl_side = 'sell' if side == 'buy' else 'buy'
            order = self.exchange.create_order(
                symbol,
                type='STOP_MARKET',
                side=sl_side,
                amount=quantity,
                params={
                    'stopPrice': stop_price,
                    'reduceOnly': True,
                    'closePosition': True,
                },
            )
            logger.info(f"Stop Loss placé {symbol} @ {stop_price:.4f}")
            return order
        except Exception as e:
            logger.error(f"Erreur place_stop_loss {symbol} : {e}")
            return None

    def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        tp_price: float,
    ) -> Optional[Dict]:
        """Place un ordre take profit."""
        try:
            tp_side = 'sell' if side == 'buy' else 'buy'
            order = self.exchange.create_order(
                symbol,
                type='TAKE_PROFIT_MARKET',
                side=tp_side,
                amount=quantity,
                params={
                    'stopPrice': tp_price,
                    'reduceOnly': True,
                    'closePosition': True,
                },
            )
            logger.info(f"Take Profit placé {symbol} @ {tp_price:.4f}")
            return order
        except Exception as e:
            logger.error(f"Erreur place_take_profit {symbol} : {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Annule un ordre."""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Ordre annulé : {order_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur cancel_order {order_id} : {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """Annule tous les ordres ouverts sur un symbole."""
        try:
            self.exchange.cancel_all_orders(symbol)
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
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Levier {symbol} : {leverage}x")
            return True
        except Exception as e:
            logger.error(f"Erreur set_leverage {symbol} : {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Retourne les ordres ouverts."""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Erreur get_open_orders : {e}")
            return []
