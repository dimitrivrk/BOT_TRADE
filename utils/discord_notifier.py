"""
Discord Notifier — envoie des notifications au bot Discord via webhook.
Messages : nouvelles bougies, signaux, trades, status périodique.
"""

import requests
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("utils.discord_notifier")

# Emojis / couleurs Discord (embed color hex)
COLOR_LONG    = 0x00C851   # vert
COLOR_SHORT   = 0xFF4444   # rouge
COLOR_CLOSE   = 0xFFBB33   # orange
COLOR_INFO    = 0x33B5E5   # bleu
COLOR_WARNING = 0xFF8800   # orange foncé
COLOR_NEUTRAL = 0xAAAAAA   # gris


class DiscordNotifier:
    """
    Envoie des embeds Discord via webhook de façon non-bloquante (thread séparé).
    Thread-safe : la file d'envoi tourne en arrière-plan.
    """

    def __init__(self, webhook_url: str, bot_name: str = "🤖 CryptoBot IA"):
        self.webhook_url = webhook_url
        self.bot_name = bot_name
        self._queue: list = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()
        logger.info("Discord notifier démarré")

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def notify_candle(
        self,
        symbol: str,
        timeframe: str,
        close: float,
        direction: int,
        confidence: float,
        regime: Optional[str] = None,
        raw_scores: Optional[dict] = None,
    ):
        """Notif à chaque bougie fermée avec le signal courant."""
        dir_label = {1: "📈 LONG", -1: "📉 SHORT", 0: "⏸ NEUTRE"}[direction]
        color = {1: COLOR_LONG, -1: COLOR_SHORT, 0: COLOR_NEUTRAL}[direction]

        fields = [
            {"name": "Prix de clôture", "value": f"**${close:,.2f}**", "inline": True},
            {"name": "Signal",          "value": dir_label,             "inline": True},
            {"name": "Confiance",       "value": f"{confidence*100:.1f}%", "inline": True},
        ]
        if regime:
            fields.append({"name": "Régime marché", "value": f"`{regime}`", "inline": True})
        if raw_scores:
            score_str = "  ".join([f"{k}: `{v:.2f}`" for k, v in raw_scores.items()])
            fields.append({"name": "Scores modèles", "value": score_str, "inline": False})

        embed = {
            "title": f"🕯 Bougie {symbol} [{timeframe}] fermée",
            "color": color,
            "fields": fields,
            "footer": {"text": self._ts()},
        }
        self._enqueue(embed)

    def notify_trade_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        usdt_amount: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        reason: str = "",
    ):
        """Notif quand un trade est ouvert."""
        emoji = "🟢" if side == "buy" else "🔴"
        label = "LONG" if side == "buy" else "SHORT"
        sl_pct = abs(entry_price - stop_loss) / entry_price * 100
        tp_pct = abs(take_profit - entry_price) / entry_price * 100
        rr    = tp_pct / sl_pct if sl_pct > 0 else 0

        embed = {
            "title": f"{emoji} ENTRÉE {label} — {symbol}",
            "color": COLOR_LONG if side == "buy" else COLOR_SHORT,
            "fields": [
                {"name": "Prix d'entrée",  "value": f"**${entry_price:,.2f}**",    "inline": True},
                {"name": "Taille",         "value": f"{quantity:.6f} ({usdt_amount:.1f} USDT)", "inline": True},
                {"name": "Confiance IA",   "value": f"{confidence*100:.1f}%",      "inline": True},
                {"name": "Stop Loss",      "value": f"${stop_loss:,.2f}  (-{sl_pct:.2f}%)", "inline": True},
                {"name": "Take Profit",    "value": f"${take_profit:,.2f}  (+{tp_pct:.2f}%)", "inline": True},
                {"name": "Risk/Reward",    "value": f"1 : {rr:.2f}",               "inline": True},
            ],
            "footer": {"text": f"{reason} • {self._ts()}"},
        }
        self._enqueue(embed)

    def notify_trade_close(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        close_price: float,
        pnl_usdt: float,
        pnl_pct: float,
        reason: str = "signal",
    ):
        """Notif quand un trade est fermé avec le P&L."""
        win = pnl_usdt >= 0
        emoji = "✅" if win else "❌"
        pnl_sign = "+" if pnl_usdt >= 0 else ""

        embed = {
            "title": f"{emoji} FERMETURE {symbol} — {reason.replace('_', ' ').upper()}",
            "color": COLOR_LONG if win else COLOR_SHORT,
            "fields": [
                {"name": "Côté",         "value": "LONG" if side == "buy" else "SHORT", "inline": True},
                {"name": "Prix entrée",  "value": f"${entry_price:,.2f}",    "inline": True},
                {"name": "Prix sortie",  "value": f"${close_price:,.2f}",    "inline": True},
                {"name": "P&L",          "value": f"**{pnl_sign}{pnl_usdt:.2f} USDT  ({pnl_sign}{pnl_pct:.2f}%)**", "inline": False},
            ],
            "footer": {"text": self._ts()},
        }
        self._enqueue(embed)

    def notify_status(
        self,
        balance_usdt: float,
        open_positions: dict,
        total_trades: int,
        win_rate: float,
        total_pnl: float,
    ):
        """Status périodique : balance + positions ouvertes + stats."""
        pos_str = ""
        if open_positions:
            for sym, pos in open_positions.items():
                side  = pos.get("side", "?")
                entry = pos.get("price", 0)
                upnl  = pos.get("unrealized_pnl", 0)
                sign  = "+" if upnl >= 0 else ""
                pos_str += f"• **{sym}** {side.upper()}  @ ${entry:,.2f}  uPnL: {sign}{upnl:.2f} USDT\n"
        else:
            pos_str = "_Aucune position ouverte_"

        pnl_sign = "+" if total_pnl >= 0 else ""
        embed = {
            "title": "📊 Status du bot",
            "color": COLOR_INFO,
            "fields": [
                {"name": "Balance",          "value": f"**{balance_usdt:.2f} USDT**", "inline": True},
                {"name": "P&L total",        "value": f"{pnl_sign}{total_pnl:.2f} USDT", "inline": True},
                {"name": "Trades / Win rate","value": f"{total_trades} trades  •  {win_rate*100:.1f}%", "inline": True},
                {"name": "Positions",        "value": pos_str, "inline": False},
            ],
            "footer": {"text": self._ts()},
        }
        self._enqueue(embed)

    def notify_warning(self, title: str, message: str):
        """Avertissement générique (bot stoppé, erreur risque, etc.)."""
        embed = {
            "title": f"⚠️ {title}",
            "description": message,
            "color": COLOR_WARNING,
            "footer": {"text": self._ts()},
        }
        self._enqueue(embed)

    def notify_start(self, pairs: list, leverage: int, testnet: bool):
        """Notif de démarrage du bot."""
        env = "🧪 TESTNET" if testnet else "🚀 MAINNET"
        embed = {
            "title": "🟢 Bot démarré",
            "color": COLOR_INFO,
            "fields": [
                {"name": "Paires",    "value": ", ".join(pairs), "inline": True},
                {"name": "Levier",   "value": f"{leverage}x",    "inline": True},
                {"name": "Environnement", "value": env,           "inline": True},
            ],
            "footer": {"text": self._ts()},
        }
        self._enqueue(embed)

    def notify_stop(self, total_pnl: float, total_trades: int):
        """Notif d'arrêt du bot."""
        sign = "+" if total_pnl >= 0 else ""
        embed = {
            "title": "🔴 Bot arrêté",
            "color": COLOR_WARNING,
            "description": f"P&L session : **{sign}{total_pnl:.2f} USDT** sur {total_trades} trade(s)",
            "footer": {"text": self._ts()},
        }
        self._enqueue(embed)

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _enqueue(self, embed: dict):
        with self._lock:
            self._queue.append({"username": self.bot_name, "embeds": [embed]})

    def _send_loop(self):
        """Thread d'envoi : dépile la queue toutes les 0.5s."""
        while True:
            payload = None
            with self._lock:
                if self._queue:
                    payload = self._queue.pop(0)
            if payload:
                try:
                    r = requests.post(self.webhook_url, json=payload, timeout=10)
                    if r.status_code == 429:          # rate limit Discord
                        retry = r.json().get("retry_after", 1)
                        time.sleep(retry / 1000 + 0.1)
                    elif r.status_code not in (200, 204):
                        logger.warning(f"Discord webhook erreur {r.status_code}: {r.text[:100]}")
                except Exception as e:
                    logger.warning(f"Discord send error: {e}")
            time.sleep(0.5)

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
