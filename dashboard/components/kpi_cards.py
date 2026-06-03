"""KPI cards component — Day P&L, open positions, win rate, capital at risk."""

import dash_bootstrap_components as dbc
from dash import html


def format_currency(value):
    if value is None or value == 0:
        return "₹0"
    sign = "-" if value < 0 else "+"
    return f"{sign}₹{abs(value):,.0f}"


def format_percent(value):
    if value is None:
        return "0%"
    return f"{value:.0%}"


def _card(label, value_elem, border_color="#2a3547"):
    return html.Div(
        children=[
            html.Div(
                label,
                style={
                    "font-size": "11px",
                    "color": "#64748b",
                    "text-transform": "uppercase",
                    "font-weight": "600",
                    "margin-bottom": "8px",
                    "letter-spacing": "0.5px",
                },
            ),
            value_elem,
        ],
        style={
            "padding": "16px",
            "background-color": "#151a24",
            "border": f"1px solid {border_color}",
            "border-radius": "6px",
            "flex": "1",
            "min-width": "160px",
        },
    )


def kpi_cards_component(bot_status=None, trades_df=None):
    if bot_status is None:
        bot_status = {}

    daily_pnl      = bot_status.get("daily_pnl", 0) or 0
    open_positions = bot_status.get("open_positions", {}) or {}
    capital_at_risk= bot_status.get("capital_at_risk", 0) or 0

    # Win rate from completed trades
    win_rate = 0
    total_trades = 0
    if trades_df is not None and not getattr(trades_df, "empty", True):
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df["pnl"] > 0])
        win_rate = wins / total_trades if total_trades > 0 else 0

    num_positions = len(open_positions)
    pnl_color = "#00ff88" if daily_pnl >= 0 else "#ff4444"

    def val(text, color, size="22px"):
        return html.Div(
            text,
            style={
                "font-family": "'Roboto Mono', monospace",
                "font-size": size,
                "font-weight": "700",
                "color": color,
                "letter-spacing": "-0.5px",
            },
        )

    cards = [
        _card("Day P&L",         val(format_currency(daily_pnl), pnl_color),                    border_color=pnl_color if daily_pnl != 0 else "#2a3547"),
        _card("Open Positions",  val(str(num_positions), "#4dabf7")),
        _card("Win Rate",        val(format_percent(win_rate), "#ffd700")),
        _card("Capital at Risk", val(f"₹{capital_at_risk:,.0f} / ₹100K", "#e2e8f0", size="14px")),
    ]

    return html.Div(
        children=cards,
        style={
            "display": "flex",
            "gap": "12px",
            "flex-wrap": "wrap",
            "padding": "12px 16px",
        },
    )
