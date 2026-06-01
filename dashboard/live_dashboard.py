"""
Live Terminal Dashboard
Rich-powered real-time view of positions, P&L, recent signals, and bot status.

Usage (standalone):
    from dashboard.live_dashboard import LiveDashboard
    dash = LiveDashboard(risk_manager, client, telegram)
    dash.run()          # blocking — press Ctrl+C to exit

Usage (embedded in main loop):
    dash = LiveDashboard(risk_manager, client)
    dash.log_signal(signal, "AI: Approved")
    dash.log_event("Stop loss hit: RELIANCE @ ₹2400")
"""

import logging
from datetime import datetime
from collections import deque
from typing import Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

_console = Console()


class LiveDashboard:
    """
    Single-file Rich dashboard. Reads state from RiskManager and KotakNeoClient.
    Call log_signal() / log_event() from the main trading loop to feed the feed panel.
    """

    MAX_FEED_LINES = 20

    def __init__(self, risk_manager, kotak_client=None, data_manager=None, refresh_rate: int = 3):
        self.risk         = risk_manager
        self.client       = kotak_client
        self.data         = data_manager
        self.refresh_rate = refresh_rate
        self._feed: deque[str] = deque(maxlen=self.MAX_FEED_LINES)
        self._live: Optional[Live] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public helpers — call from trading loop
    # ─────────────────────────────────────────────────────────────────────────

    def log_signal(self, signal, ai_note: str = ""):
        emoji = "🟢" if signal.action == "BUY" else "🔴"
        conf  = f"{signal.confidence:.0%}"
        line  = (
            f"{self._now()} {emoji} {signal.action} {signal.symbol}"
            f" ₹{signal.price:,.2f}  conf={conf}"
            f" [{signal.strategy}]"
        )
        if ai_note:
            line += f"  AI: {ai_note[:60]}"
        self._feed.append(line)

    def log_event(self, message: str):
        self._feed.append(f"{self._now()} {message}")

    # ─────────────────────────────────────────────────────────────────────────
    # Blocking runner
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Block and refresh until Ctrl+C."""
        layout = self._build_layout()
        try:
            with Live(layout, console=_console, refresh_per_second=self.refresh_rate,
                      screen=True) as live:
                self._live = live
                import time
                while True:
                    self._update_layout(layout)
                    time.sleep(1 / self.refresh_rate)
        except KeyboardInterrupt:
            pass
        finally:
            self._live = None

    def render_once(self) -> Layout:
        """Return a fully populated Layout (for embedding in a larger UI)."""
        layout = self._build_layout()
        self._update_layout(layout)
        return layout

    # ─────────────────────────────────────────────────────────────────────────
    # Layout construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header",    size=3),
            Layout(name="stats",     size=5),
            Layout(name="main",      ratio=1),
            Layout(name="footer",    size=3),
        )
        layout["main"].split_row(
            Layout(name="positions", ratio=3),
            Layout(name="feed",      ratio=2),
        )
        return layout

    def _update_layout(self, layout: Layout):
        layout["header"].update(self._render_header())
        layout["stats"].update(self._render_stats())
        layout["positions"].update(self._render_positions())
        layout["feed"].update(self._render_feed())
        layout["footer"].update(self._render_footer())

    # ─────────────────────────────────────────────────────────────────────────
    # Panels
    # ─────────────────────────────────────────────────────────────────────────

    def _render_header(self) -> Panel:
        now     = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        paper   = "[yellow]PAPER TRADING[/yellow]"
        if self.client and not getattr(self.client, "paper_trading", True):
            paper = "[bold red]LIVE TRADING[/bold red]"
        title = Align.center(
            Text.from_markup(f"[bold cyan]AlgoTrader Dashboard[/bold cyan]  |  {now}  |  {paper}")
        )
        return Panel(title, style="bold blue")

    def _render_stats(self) -> Panel:
        portfolio = self.risk.get_portfolio_summary()
        pnl       = portfolio.get("daily_pnl", 0)
        trades    = portfolio.get("daily_trades", 0)
        n_open    = portfolio.get("position_count", 0)
        at_risk   = portfolio.get("capital_at_risk", 0)

        pnl_color = "green" if pnl >= 0 else "red"
        pnl_sign  = "+" if pnl >= 0 else ""

        table = Table.grid(expand=True, padding=(0, 4))
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")

        table.add_row(
            f"[bold]Day P&L[/bold]\n[{pnl_color}]₹{pnl_sign}{pnl:,.0f}[/{pnl_color}]",
            f"[bold]Trades Today[/bold]\n[cyan]{trades}[/cyan]",
            f"[bold]Open Positions[/bold]\n[cyan]{n_open} / {self.risk.config.max_open_positions}[/cyan]",
            f"[bold]Capital at Risk[/bold]\n[yellow]₹{at_risk:,.0f}[/yellow]",
        )
        return Panel(table, title="Portfolio", border_style="green" if pnl >= 0 else "red")

    def _render_positions(self) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True,
                      header_style="bold magenta")
        table.add_column("Symbol",  style="bold", min_width=12)
        table.add_column("Action",  justify="center", width=6)
        table.add_column("Entry",   justify="right", width=10)
        table.add_column("SL",      justify="right", width=10)
        table.add_column("Target",  justify="right", width=10)
        table.add_column("P&L",     justify="right", width=10)
        table.add_column("Status",  justify="center", width=10)

        positions = self.risk.get_portfolio_summary().get("open_positions", {})

        if not positions:
            table.add_row("—", "—", "—", "—", "—", "—", "[dim]No open positions[/dim]")
        else:
            for sym, pos in positions.items():
                entry   = pos["entry_price"]
                sl      = pos["stop_loss"]
                target  = pos["target"]
                qty     = pos["quantity"]

                # Try to get current price if data manager attached
                current = entry
                if self.data:
                    try:
                        q = self.data.get_live_quote(sym, pos.get("exchange", "nse_cm"))
                        if q:
                            current = q["ltp"]
                    except Exception:
                        pass

                pnl_val = (current - entry) * qty if pos["action"] == "BUY" else (entry - current) * qty
                pnl_color = "green" if pnl_val >= 0 else "red"
                pnl_sign  = "+" if pnl_val >= 0 else ""

                sl_dist_pct = abs(current - sl) / entry * 100
                if sl_dist_pct < 0.5:
                    status = "[bold red]Near SL![/bold red]"
                elif pos["action"] == "BUY" and current >= target * 0.97:
                    status = "[bold green]Near Target[/bold green]"
                else:
                    status = "[dim]Hold[/dim]"

                action_color = "green" if pos["action"] == "BUY" else "red"
                table.add_row(
                    sym,
                    f"[{action_color}]{pos['action']}[/{action_color}]",
                    f"₹{entry:,.2f}",
                    f"₹{sl:,.2f}",
                    f"₹{target:,.2f}",
                    f"[{pnl_color}]{pnl_sign}₹{pnl_val:,.0f}[/{pnl_color}]",
                    status,
                )

        return Panel(table, title="Open Positions", border_style="blue")

    def _render_feed(self) -> Panel:
        lines = list(self._feed) or ["[dim]No signals yet. Waiting for market...[/dim]"]
        text  = "\n".join(reversed(lines))  # newest at top
        return Panel(
            Text.from_markup(text),
            title="Signal Feed",
            border_style="yellow",
        )

    def _render_footer(self) -> Panel:
        session_age = ""
        if self.client and getattr(self.client, "session_time", None):
            mins = int((datetime.now() - self.client.session_time).total_seconds() / 60)
            session_age = f"Session: {mins}m"

        auth_status = ""
        if self.client:
            auth_status = (
                "[green]Authenticated[/green]"
                if self.client.is_authenticated()
                else "[red]Not authenticated[/red]"
            )

        footer_text = Align.center(
            Text.from_markup(
                f"[dim]{auth_status}  |  {session_age}  |  Press Ctrl+C to stop[/dim]"
            )
        )
        return Panel(footer_text, style="dim")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%H:%M:%S")
