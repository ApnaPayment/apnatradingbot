"""
Telegram Alert Bot
Sends real-time trading alerts, P&L updates, and AI summaries.
"""

import html
import os
import logging
import requests
from datetime import datetime
from core.risk_manager import TradeSignal


def _esc(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html.escape(str(text))

logger = logging.getLogger(__name__)


class TelegramAlerter:
    """Send trading alerts via Telegram bot."""

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self):
        self.token    = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id  = os.getenv("TELEGRAM_CHAT_ID")
        chat_id_2     = os.getenv("TELEGRAM_CHAT_ID_2", "")
        # Build list of all recipients
        self.chat_ids = [c for c in [self.chat_id, chat_id_2] if c]
        self.enabled  = bool(self.token and self.chat_ids)
        if not self.enabled:
            logger.warning("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

    def send(self, message: str, silent: bool = False) -> bool:
        """Send a plain text message to all configured chat IDs."""
        if not self.enabled:
            logger.info(f"[TELEGRAM DISABLED] {message}")
            return False
        url = self.BASE_URL.format(token=self.token)
        success = True
        for cid in self.chat_ids:
            try:
                payload = {
                    "chat_id": cid,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_notification": silent,
                }
                r = requests.post(url, json=payload, timeout=10)
                r.raise_for_status()
            except Exception as e:
                logger.error(f"Telegram send failed for {cid}: {e}")
                success = False
        return success

    # ─────────────────────────────────────────────────────────────────────────
    # Formatted alert types
    # ─────────────────────────────────────────────────────────────────────────

    def alert_trade_signal(self, signal: TradeSignal, ai_reasoning: str = ""):
        emoji = "🟢" if signal.action == "BUY" else "🔴"
        rr = ""
        if signal.stop_loss and signal.target and signal.price:
            risk   = abs(signal.price - signal.stop_loss)
            reward = abs(signal.target - signal.price)
            if risk > 0:
                rr = f"\n⚖️ R:R = 1:{reward/risk:.1f}"

        msg = (
            f"{emoji} <b>{signal.action} SIGNAL — {_esc(signal.symbol)}</b>\n"
            f"{'─' * 28}\n"
            f"💰 Entry:     ₹{signal.price:,.2f}\n"
            f"🛑 Stop Loss: ₹{signal.stop_loss:,.2f}\n"
            f"🎯 Target:    ₹{signal.target:,.2f}\n"
            f"📊 Strategy:  {_esc(signal.strategy.title())}\n"
            f"🔮 Confidence: {signal.confidence:.0%}\n"
            f"{rr}\n\n"
            f"<i>{_esc(signal.reasoning[:200])}</i>"
        )
        if ai_reasoning:
            msg += f"\n\n🤖 <b>AI:</b> {_esc(ai_reasoning[:200])}"

        self.send(msg)

    def alert_ai_veto(self, signal, reasoning: str, concerns: list,
                      tools_used: list, live_price: float = None):
        """
        Structured AI veto alert.
        Format inspired by user request — concise, bullet-pointed, at-a-glance.
        """
        strat_map = {
            "momentum":       "Momentum",
            "mean_reversion": "Mean Reversion",
            "options":        "Options",
        }
        strategy    = strat_map.get(signal.strategy, signal.strategy.title())
        conf_pct    = f"{signal.confidence:.0%}"
        entry_price = signal.price
        ltp_line    = f"📍 Current:       ₹{live_price:,.2f}\n" if live_price else ""

        # Build bullet points from concerns (max 4), or fall back to reasoning snippet
        if concerns:
            bullets = "\n".join(f"  • {_esc(c)}" for c in concerns[:4])
        else:
            # Chunk reasoning into short bullets
            sentences = [s.strip() for s in reasoning.replace(". ", ".|").split("|") if s.strip()]
            bullets = "\n".join(f"  • {_esc(s)}" for s in sentences[:3])

        tools_line = f"🔧 Verified via: {', '.join(tools_used)}\n" if tools_used else ""

        msg = (
            f"🚫 <b>AI Veto — {_esc(signal.symbol.replace('-EQ', ''))}</b>\n"
            f"{'─' * 28}\n"
            f"📊 Strategy:      {strategy}\n"
            f"🎯 Confidence:    {conf_pct}\n"
            f"{'─' * 28}\n"
            f"<b>Reason:</b>\n{bullets}\n"
            f"{'─' * 28}\n"
            f"{ltp_line}"
            f"📌 Planned entry: ₹{entry_price:,.2f}\n"
            f"{tools_line}"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send(msg)

    def alert_order_placed(self, symbol: str, action: str, quantity: int,
                            price: float, order_no: str):
        emoji = "✅" if action == "BUY" else "📤"
        msg = (
            f"{emoji} <b>Order Placed</b>\n"
            f"{'─' * 28}\n"
            f"📌 {action} {quantity} × {symbol}\n"
            f"💰 Price: ₹{price:,.2f}\n"
            f"🔢 Order No: <code>{order_no}</code>\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send(msg)

    def alert_stop_loss_hit(self, symbol: str, entry_price: float,
                             exit_price: float, pnl: float, quantity: int):
        msg = (
            f"🛑 <b>Stop Loss Hit — {symbol}</b>\n"
            f"{'─' * 28}\n"
            f"📥 Entry: ₹{entry_price:,.2f}\n"
            f"📤 Exit:  ₹{exit_price:,.2f}\n"
            f"📦 Qty:   {quantity}\n"
            f"{'🔴' if pnl < 0 else '🟢'} P&L: ₹{pnl:,.0f}"
        )
        self.send(msg)

    def alert_target_hit(self, symbol: str, entry_price: float,
                          exit_price: float, pnl: float, quantity: int):
        msg = (
            f"🎯 <b>Target Hit — {symbol}</b>\n"
            f"{'─' * 28}\n"
            f"📥 Entry: ₹{entry_price:,.2f}\n"
            f"📤 Exit:  ₹{exit_price:,.2f}\n"
            f"📦 Qty:   {quantity}\n"
            f"🟢 P&L: ₹{pnl:,.0f}\n"
            f"🎉 Profit booked!"
        )
        self.send(msg)

    def alert_daily_summary(self, portfolio: dict, ai_advice=None,
                             session_stats: dict = None):
        """
        Full EOD report — mobile-friendly, no raw JSON, all sections structured.
        ai_advice: dict with keys portfolio_health / key_risks / tomorrow_focus / summary
        """
        pnl       = portfolio.get("daily_pnl", 0)
        trades    = portfolio.get("daily_trades", 0)
        n_open    = portfolio.get("position_count", 0)
        at_risk   = portfolio.get("capital_at_risk", 0)
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        date_str  = datetime.now().strftime("%d %b %Y")
        s         = session_stats or {}

        def _fmt(val, fmt=".1f", suffix="", na="N/A"):
            if val is None:
                return na
            if val == float("inf"):
                return "∞"
            return f"{val:{fmt}}{suffix}"

        # ── Section 1: P&L ──────────────────────────────────────────────────
        lines = [
            f"📊 <b>EOD Report — {date_str}</b>",
            f"{'─' * 30}",
            f"{pnl_emoji} <b>Day P&L</b>       ₹{pnl:+,.0f}",
            f"🔁 Trades          {trades}",
            f"📂 Open positions  {n_open}",
        ]
        if at_risk > 0:
            lines.append(f"⚠️ Capital at risk  ₹{at_risk:,.0f}")

        # ── Section 2: Bot Operations ────────────────────────────────────────
        ws_icon  = "🟢" if s.get("ws_connected") else "🔴"
        coverage = s.get("market_coverage", "—")
        cov_icon = "✅" if coverage == "YES" else ("⚠️" if coverage == "PARTIAL" else "❌")
        lines += [
            f"{'─' * 30}",
            f"⚙️ <b>Bot Operations</b>",
            f"🕐 Started          {s.get('session_start', '—')}",
            f"{cov_icon} Market coverage   {coverage}",
            f"🔍 First scan       {s.get('first_scan', '—')}",
            f"🔍 Last scan        {s.get('last_scan', '—')}",
            f"🔄 Scan cycles      {s.get('scan_cycles', 0)}",
            f"{ws_icon} WebSocket         {'Connected' if s.get('ws_connected') else 'Disconnected'}",
            f"⏱ Uptime           {_fmt(s.get('uptime_pct'), '.0f', '%')}",
        ]

        # ── Section 3: Signal Analytics ─────────────────────────────────────
        approved     = s.get("ai_approved", 0)
        rejected     = s.get("ai_rejected", 0)
        appr_rate    = _fmt(s.get("approval_rate"), ".0f", "%")
        good_v       = s.get("good_vetoes", 0)
        missed_v     = s.get("missed_opportunities", 0)
        veto_acc     = _fmt(s.get("veto_accuracy"), ".0f", "%")
        lines += [
            f"{'─' * 30}",
            f"📡 <b>Signal Analytics</b>",
            f"   Generated       {s.get('signals_today', 0)}",
            f"   ✅ AI approved   {approved}",
            f"   ❌ AI rejected   {rejected}",
            f"   Approval rate   {appr_rate}",
        ]
        if good_v + missed_v > 0:
            lines += [
                f"   Good vetoes     {good_v}  (saved losses)",
                f"   Missed opps     {missed_v}  (would have won)",
                f"   Veto accuracy   {veto_acc}",
            ]

        # ── Section 4: Performance Metrics ───────────────────────────────────
        win_str  = _fmt(s.get("win_rate"), ".1f", "%")
        pf_str   = _fmt(s.get("profit_factor"), ".2f")
        dd_str   = f"₹{s.get('max_drawdown', 0):,.0f}" if s.get("max_drawdown", 0) > 0 else "₹0"
        aw_str   = f"₹{s.get('avg_win', 0):+,.0f}"  if s.get("avg_win")  is not None else "N/A"
        al_str   = f"₹{s.get('avg_loss', 0):,.0f}"  if s.get("avg_loss") is not None else "N/A"
        rr_str   = _fmt(s.get("rr_ratio"), ".2f", ":1")
        lines += [
            f"{'─' * 30}",
            f"📈 <b>Performance</b>",
            f"   Win rate        {win_str}",
            f"   Profit factor   {pf_str}",
            f"   Avg win         {aw_str}",
            f"   Avg loss        {al_str}",
            f"   Risk/Reward     {rr_str}",
            f"   Max drawdown    {dd_str}",
        ]

        # ── Section 5: Strategy Breakdown ────────────────────────────────────
        strat_stats = s.get("strategy_stats", {})
        strat_map   = {"momentum": "Momentum", "mean_reversion": "Mean Rev", "options": "Options"}
        active_strats = [(k, v) for k, v in strat_stats.items() if v.get("trades", 0) > 0]
        if active_strats:
            lines.append(f"{'─' * 30}")
            lines.append(f"🧠 <b>Strategy Breakdown</b>")
            for key, st in active_strats:
                wr  = _fmt(st.get("win_rate"), ".0f", "%")
                pnl_s = f"₹{st['pnl']:+,.0f}"
                lines.append(f"   {strat_map.get(key, key):<10}  {st['trades']}T  WR:{wr}  {pnl_s}")
        else:
            lines += [
                f"{'─' * 30}",
                f"🧠 <b>Strategy Breakdown</b>",
                f"   No trades executed today",
            ]

        # ── Section 6: AI Advice (structured, no JSON) ───────────────────────
        if isinstance(ai_advice, dict) and ai_advice:
            lines.append(f"{'─' * 30}")
            lines.append(f"🤖 <b>AI Advice</b>")
            health = ai_advice.get("portfolio_health", "")
            if health:
                lines.append(f"   Health: {health}")
            risks = ai_advice.get("key_risks", [])
            if risks:
                lines.append("   <b>Key Risks:</b>")
                for r in risks[:3]:
                    lines.append(f"   • {r}")
            focus = ai_advice.get("tomorrow_focus", [])
            if focus:
                lines.append("   <b>Tomorrow:</b>")
                for f_ in focus[:3]:
                    lines.append(f"   → {f_}")
            summary = ai_advice.get("summary", "")
            if summary:
                lines.append(f"   💬 {summary}")

        self.send("\n".join(lines))

    def alert_weekly_summary(self, report: dict):
        """
        Friday EOW report sent to Telegram.

        report keys:
            week_label          str   e.g. "26 May – 30 May 2026"
            week_pnl            float
            total_trades        int
            win_rate            float  0–1
            profit_factor       float
            avg_winner          float
            avg_loser           float
            best_symbol         str
            best_symbol_pnl     float
            worst_symbol        str
            worst_symbol_pnl    float
            strategy_breakdown  list of dicts {name, trades, wr, pnl}
            ai_decisions_total  int
            ai_approved         int
            ai_vetoed           int
            ai_accuracy         float | None  (fraction of good decisions)
            cumulative_pnl      float
            milestone_trades    int   (total closed since bot started)
            milestone_target    int   (100)
            journal_days        list of dicts {date, day_pnl, regime}
        """
        def _fmt(v, fmt=".1f", suffix="", na="N/A"):
            if v is None:
                return na
            if v == float("inf"):
                return "∞"
            return f"{v:{fmt}}{suffix}"

        def _pf_str(pf):
            if pf == float("inf"):
                return "∞"
            if pf is None:
                return "N/A"
            return f"{pf:.3f}"

        pnl       = report.get("week_pnl", 0)
        cum_pnl   = report.get("cumulative_pnl", 0)
        pnl_e     = "🟢" if pnl >= 0 else "🔴"
        cum_e     = "🟢" if cum_pnl >= 0 else "🔴"
        n         = report.get("total_trades", 0)
        wr        = report.get("win_rate", 0)
        pf        = report.get("profit_factor", 0)
        aw        = report.get("avg_winner", 0)
        al        = report.get("avg_loser", 0)
        milestone = report.get("milestone_trades", 0)
        target    = report.get("milestone_target", 100)
        remaining = max(0, target - milestone)

        # Milestone progress bar (20 chars)
        filled  = min(milestone, target)
        bar_len = 20
        filled_b = int(filled / target * bar_len)
        bar = "█" * filled_b + "░" * (bar_len - filled_b)

        lines = [
            f"📅 <b>Weekly Summary — {report.get('week_label', '')}</b>",
            f"{'─' * 34}",
            f"",
            f"💰 <b>P&L</b>",
            f"  {pnl_e} Week P&L       ₹{pnl:+,.0f}",
            f"  {cum_e} Cumulative     ₹{cum_pnl:+,.0f}",
            f"",
            f"📊 <b>Trades</b>",
            f"  🔁 Total           {n}",
            f"  ✅ Win rate        {wr:.1%}",
            f"  ⚖️  Profit factor   {_pf_str(pf)}",
            f"  📈 Avg winner      ₹{aw:+,.0f}",
            f"  📉 Avg loser       ₹{al:,.0f}",
        ]

        # Best / worst symbol
        best_sym = report.get("best_symbol")
        worst_sym = report.get("worst_symbol")
        if best_sym or worst_sym:
            lines += [f"", f"🏆 <b>Symbol Performance</b>"]
            if best_sym:
                lines.append(
                    f"  🥇 Best   {best_sym:<14} ₹{report.get('best_symbol_pnl', 0):+,.0f}"
                )
            if worst_sym:
                lines.append(
                    f"  💀 Worst  {worst_sym:<14} ₹{report.get('worst_symbol_pnl', 0):+,.0f}"
                )

        # Strategy breakdown
        strategies = report.get("strategy_breakdown", [])
        if strategies:
            lines += [f"", f"🧠 <b>Strategy Breakdown</b>"]
            for s in strategies:
                wr_s  = f"{s.get('wr', 0):.0%}"
                pnl_s = f"₹{s.get('pnl', 0):+,.0f}"
                name  = s.get("name", "?")[:12]
                lines.append(
                    f"  {name:<14} {s.get('trades', 0):>3}T  WR:{wr_s}  {pnl_s}"
                )

        # AI decision accuracy
        ai_total  = report.get("ai_decisions_total", 0)
        ai_app    = report.get("ai_approved", 0)
        ai_veto   = report.get("ai_vetoed", 0)
        ai_acc    = report.get("ai_accuracy")
        lines += [f"", f"🤖 <b>AI Decision Log</b>"]
        lines.append(f"  📋 Evaluated       {ai_total}")
        lines.append(f"  ✅ Approved        {ai_app}  |  ❌ Vetoed  {ai_veto}")
        if ai_acc is not None:
            acc_e = "✅" if ai_acc >= 0.6 else "⚠️"
            lines.append(f"  {acc_e} Veto accuracy    {ai_acc:.0%}")
        else:
            lines.append(f"  ⏳ Accuracy pending (outcomes not resolved yet)")

        # Daily journal strip (mini calendar)
        days = report.get("journal_days", [])
        if days:
            lines += [f"", f"📆 <b>Day-by-Day</b>"]
            for d in days:
                dp    = d.get("day_pnl", 0)
                emoji = "🟢" if dp > 0 else ("⚫" if dp == 0 else "🔴")
                regime = (d.get("regime") or "?")[:8]
                lines.append(
                    f"  {emoji} {d.get('date','?')}  ₹{dp:+,.0f}  {regime}"
                )

        # Milestone progress
        lines += [
            f"",
            f"🎯 <b>Milestone Progress</b>  ({milestone}/{target} trades)",
            f"  [{bar}] {milestone/target:.0%}",
        ]
        if milestone >= target:
            lines.append(f"  ✅ <b>MILESTONE REACHED — Re-evaluate strategies now!</b>")
        else:
            lines.append(f"  ⏳ {remaining} more trades to re-evaluation")

        lines += [f"", f"{'─' * 34}", f"📱 <i>AlgoTrader | PAPER mode</i>"]

        self.send("\n".join(lines))

    def alert_error(self, error_msg: str, context: str = ""):
        msg = (
            f"⚠️ <b>System Alert</b>\n"
            f"{'─' * 28}\n"
            f"Context: {context}\n"
            f"Error: <code>{error_msg[:300]}</code>\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send(msg)

    def alert_market_open(self):
        self.send("🔔 <b>Market Open</b> — Bot is live and scanning. Good luck! 🚀", silent=True)

    def alert_market_close(self, portfolio: dict = None):
        self.alert_daily_summary(portfolio or {}, {}, {})

    def alert_fo_trade(self, symbol: str, action: str, qty: int,
                       premium: float, strike: int, expiry, option_type: str,
                       underlying: str, days_to_expiry: int,
                       exit_reason: str = None, paper: bool = True,
                       stop_loss: float = None, target: float = None):
        """Alert for F&O options trade entry or exit."""
        lot_cost = premium * qty
        is_sell  = (action == "SELL")

        if exit_reason:
            reason_map = {
                # SELL (short option) exit reasons
                "target_70pct":  "🎯 Target Hit — premium decayed 70% (kept 70% of credit)",
                "sl_100pct":     "🛑 Stop Loss — premium doubled (100% loss on credit)",
                # BUY (long option) exit reasons
                "target_50pct":  "🎯 Target Hit (50% gain)",
                "sl_30pct":      "🛑 Stop Loss (30% loss)",
                # Common
                "expiry_exit":   "📅 Expiry Exit (1 day left)",
                "eod_squareoff": "🔔 EOD Square-off",
            }
            reason_label = reason_map.get(exit_reason, exit_reason)
            msg = (
                f"📤 <b>F&O EXIT — {symbol}</b>\n"
                f"{'─' * 28}\n"
                f"💰 Exit Premium: ₹{premium:.2f} × {qty} = ₹{lot_cost:,.0f}\n"
                f"📌 {reason_label}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
        else:
            ot_emoji = "📈" if option_type == "CE" else "📉"
            mode_tag = "PAPER " if paper else ""

            if is_sell:
                # Short option: SL = premium rises (bad), Target = premium falls (good)
                sl_price  = stop_loss if stop_loss else round(premium * 2.00, 2)   # doubles → stop
                tgt_price = target    if target    else round(premium * 0.30, 2)   # 70% decay → profit
                sl_label  = f"₹{sl_price:.2f} (buy back if premium doubles)"
                tgt_label = f"₹{tgt_price:.2f} (buy back when 70% decayed — keep ₹{(premium-tgt_price)*qty:,.0f})"
                credit_note = f"💳 Credit received: ₹{lot_cost:,.0f} (profit if premium decays)\n"
            else:
                sl_price  = stop_loss if stop_loss else round(premium * 0.70, 2)
                tgt_price = target    if target    else round(premium * 1.50, 2)
                sl_label  = f"₹{sl_price:.2f} (30% loss)"
                tgt_label = f"₹{tgt_price:.2f} (50% gain)"
                credit_note = ""

            msg = (
                f"{ot_emoji} <b>{mode_tag}{action} {option_type} — {underlying} {strike} {option_type}</b>\n"
                f"{'─' * 28}\n"
                f"🏷 Symbol: <code>{symbol}</code>\n"
                f"📅 Expiry: {expiry} ({days_to_expiry} days)\n"
                f"💰 Premium: ₹{premium:.2f} × {qty} = ₹{lot_cost:,.0f}\n"
                f"{credit_note}"
                f"🛡 SL: {sl_label}\n"
                f"🎯 Target: {tgt_label}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
        self.send(msg)

    def test(self) -> bool:
        """Send a test message to verify bot is working."""
        return self.send("✅ <b>Algo Trader Bot</b> connected successfully!\nReady to trade. 🚀")
