"""Header component — status bar showing bot state, regime, mode, time."""

import dash_bootstrap_components as dbc
from dash import html
from datetime import datetime, timedelta


def _next_cycle_str(last_cycle_str):
    """Compute next cycle time and seconds remaining from last_cycle timestamp."""
    try:
        last = datetime.fromisoformat(str(last_cycle_str))
        nxt  = last + timedelta(minutes=5)
        now  = datetime.now()
        secs = int((nxt - now).total_seconds())
        if secs < 0:
            return "now", 0
        mins, s = divmod(secs, 60)
        label = f"{mins}m {s:02d}s" if mins else f"{s}s"
        return nxt.strftime("%H:%M"), secs
    except Exception:
        return "—", 999


def header_component(bot_status=None):
    if bot_status is None:
        bot_status = {}

    status    = bot_status.get("status", "stopped")
    is_paper  = bot_status.get("paper_trading", 1)
    regime    = bot_status.get("market_regime", "unknown") or "unknown"
    ai_conf   = bot_status.get("ai_regime_confidence", 0) or 0
    ai_risk   = (bot_status.get("ai_regime_risk") or "medium").lower()
    ai_suggestion = bot_status.get("ai_regime_suggestion") or ""
    last_cycle = bot_status.get("last_cycle")

    # Next cycle countdown
    next_time, secs_left = _next_cycle_str(last_cycle)
    if secs_left == 0:
        cycle_label = "⚡ scanning now"
        cycle_color = "#00ff88"
    elif secs_left < 60:
        cycle_label = f"⏱ next scan {secs_left}s"
        cycle_color = "#ffd700"
    else:
        mins, s = divmod(secs_left, 60)
        cycle_label = f"⏱ next scan {mins}m {s:02d}s"
        cycle_color = "#64748b"

    status_symbol = "🟢" if status == "running" else "🔴"
    mode_text  = "PAPER" if is_paper else "LIVE"
    mode_color = "#4dabf7" if is_paper else "#ff6b6b"

    # Clean regime display
    regime_display = "—" if regime.lower() in ("unknown", "", "none") else regime.upper()
    regime_color_map = {
        "trending_up":    "#16a34a",
        "trending_down":  "#dc2626",
        "ranging":        "#2563eb",
        "high_volatility":"#d97706",
        "breakout":       "#9333ea",
    }
    regime_color = regime_color_map.get(regime.lower(), "#64748b")
    regime_bg_map = {
        "trending_up":    "rgba(22,163,74,0.15)",
        "trending_down":  "rgba(220,38,38,0.15)",
        "ranging":        "rgba(37,99,235,0.15)",
        "high_volatility":"rgba(217,119,6,0.15)",
        "breakout":       "rgba(168,85,247,0.15)",
    }
    regime_bg = regime_bg_map.get(regime.lower(), "rgba(100,116,139,0.15)")

    current_time = datetime.now().strftime("%H:%M:%S IST")

    return html.Div(
        children=[
            html.Div(
                style={
                    "padding": "10px 16px",
                    "background-color": "#151a24",
                    "border-bottom": "1px solid #2a3547",
                    "font-family": "'Roboto Mono', monospace",
                    "display": "flex",
                    "justify-content": "space-between",
                    "align-items": "center",
                },
                children=[
                    # Left: Bot status
                    html.Div(
                        children=[
                            html.Span(status_symbol, style={"margin-right": "8px", "font-size": "14px"}),
                            html.Span(
                                "AlgoTrader",
                                style={
                                    "font-weight": "700",
                                    "font-size": "13px",
                                    "text-transform": "uppercase",
                                    "letter-spacing": "0.5px",
                                    "color": "#e2e8f0",
                                },
                            ),
                        ],
                        style={"display": "flex", "align-items": "center"},
                    ),
                    # Center: Mode, regime, AI confidence
                    html.Div(
                        children=[
                            html.Span(
                                mode_text,
                                style={
                                    "padding": "3px 8px",
                                    "background-color": "rgba(77,171,247,0.2)" if is_paper else "rgba(255,107,107,0.2)",
                                    "color": mode_color,
                                    "border-radius": "3px",
                                    "font-size": "11px",
                                    "font-weight": "600",
                                    "letter-spacing": "0.5px",
                                },
                            ),
                            html.Span(
                                regime_display,
                                title=ai_suggestion,
                                style={
                                    "padding": "3px 8px",
                                    "background-color": regime_bg,
                                    "color": regime_color,
                                    "border-radius": "3px",
                                    "font-size": "11px",
                                    "font-weight": "600",
                                    "letter-spacing": "0.5px",
                                    "cursor": "help",
                                },
                            ),
                            *(
                                [html.Span(
                                    f"AI {ai_conf:.0%}",
                                    style={
                                        "padding": "3px 6px",
                                        "background-color": {
                                            "low":    "rgba(22,163,74,0.1)",
                                            "medium": "rgba(217,119,6,0.1)",
                                            "high":   "rgba(220,38,38,0.1)",
                                        }.get(ai_risk, "rgba(100,116,139,0.1)"),
                                        "color": {
                                            "low":    "#16a34a",
                                            "medium": "#d97706",
                                            "high":   "#dc2626",
                                        }.get(ai_risk, "#64748b"),
                                        "border-radius": "3px",
                                        "font-size": "10px",
                                        "font-weight": "600",
                                    },
                                )] if ai_conf > 0 else []
                            ),
                        ],
                        style={"display": "flex", "align-items": "center", "gap": "8px"},
                    ),
                    # Right: Cycle countdown + Time
                    html.Div(
                        children=[
                            html.Span(
                                cycle_label,
                                style={
                                    "font-size": "10px",
                                    "color": cycle_color,
                                    "font-family": "'Roboto Mono', monospace",
                                    "padding": "2px 6px",
                                    "background-color": "rgba(100,116,139,0.1)",
                                    "border-radius": "3px",
                                },
                            ),
                            html.Span(
                                current_time,
                                style={"font-size": "11px", "color": "#64748b"},
                            ),
                        ],
                        style={"display": "flex", "align-items": "center", "gap": "10px", "text-align": "right"},
                    ),
                ],
            ),
        ],
        id="header-container",
        style={"background-color": "#151a24", "margin": "0", "padding": "0"},
    )
