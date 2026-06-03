"""Risk panel component — Capital gauges, daily loss meter, VIX, FII/DII, circuit breaker, calendar."""

from dash import html, dcc
import plotly.graph_objects as go

_CARD = {"background-color": "#151a24", "border": "1px solid #2a3547",
         "border-radius": "6px", "padding": "14px"}


def create_radial_gauge(value, max_value, label):
    pct = min(100, max(0, (value / max_value * 100) if max_value > 0 else 0))
    color = "#00ff88" if pct < 60 else ("#ffd700" if pct < 80 else "#ff4444")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": label, "font": {"size": 12, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2a3547"},
            "bar": {"color": color},
            "bgcolor": "#0d0f14",
            "bordercolor": "#2a3547",
            "steps": [
                {"range": [0,  60], "color": "rgba(0,255,136,0.08)"},
                {"range": [60, 80], "color": "rgba(255,215,0,0.08)"},
                {"range": [80,100], "color": "rgba(255,68,68,0.08)"},
            ],
            "threshold": {"line": {"color": "#2a3547", "width": 2}, "thickness": 0.75, "value": 100},
        },
        number={"suffix": "%", "font": {"size": 18, "color": color}},
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151a24",
        plot_bgcolor="#151a24",
        height=220,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        font={"size": 11, "family": "'Roboto Mono', monospace", "color": "#94a3b8"},
    )
    return dcc.Graph(figure=fig, style={"height": "220px"}, config={"displayModeBar": False})


def _vix_bar(vix):
    if vix is None:
        return html.Div("India VIX: unavailable",
                        style={"color": "#64748b", "font-size": "11px", **_CARD})
    pct = min(100, vix / 30 * 100)
    color = "#16a34a" if vix < 15 else ("#d97706" if vix < 20 else ("#dc2626" if vix < 25 else "#7f1d1d"))
    label = "Normal" if vix < 15 else ("Elevated" if vix < 20 else ("High" if vix < 25 else "EXTREME"))

    return html.Div(
        children=[
            html.Div(children=[
                html.Span("India VIX", style={"font-size": "11px", "color": "#64748b",
                                              "text-transform": "uppercase", "font-weight": "600"}),
                html.Span(f"{vix:.2f}  —  {label}",
                          style={"font-size": "14px", "font-weight": "700", "color": color,
                                 "font-family": "'Roboto Mono', monospace"}),
            ], style={"display": "flex", "justify-content": "space-between",
                      "align-items": "center", "margin-bottom": "8px"}),
            html.Div(children=[html.Div(style={
                "width": f"{pct}%", "height": "8px", "background-color": color,
                "border-radius": "2px", "transition": "width 0.3s ease",
            })], style={"width": "100%", "height": "8px", "background-color": "#2a3547",
                        "border-radius": "2px", "overflow": "hidden", "margin-bottom": "6px"}),
            html.Div(children=[
                html.Span("0",        style={"font-size": "9px", "color": "#64748b"}),
                html.Span("15 OK",    style={"font-size": "9px", "color": "#16a34a"}),
                html.Span("20 High",  style={"font-size": "9px", "color": "#d97706"}),
                html.Span("25 Ext",   style={"font-size": "9px", "color": "#dc2626"}),
                html.Span("30+",      style={"font-size": "9px", "color": "#64748b"}),
            ], style={"display": "flex", "justify-content": "space-between"}),
        ],
        style=_CARD,
    )


def _circuit_breaker_strip(bot_status):
    try:
        from core.circuit_breaker import CircuitBreakerConfig
        cfg = CircuitBreakerConfig()
        cb_info   = (bot_status or {}).get("circuit_breaker", {})
        state_val = cb_info.get("state", "normal")
        thresholds = (f"L1={cfg.level1_pct}%  L2={cfg.level2_pct}%  "
                      f"L3={cfg.level3_pct}%  (of ₹{cfg.max_capital:,.0f})")
    except Exception:
        state_val  = "normal"
        thresholds = ""

    COLOR = {"normal": "#16a34a", "caution": "#d97706", "halt": "#dc2626", "close": "#7f1d1d"}
    BG    = {"normal": "rgba(22,163,74,0.1)", "caution": "rgba(217,119,6,0.1)",
             "halt":   "rgba(220,38,38,0.1)", "close":   "rgba(127,29,29,0.15)"}
    EMOJI = {"normal": "🟢", "caution": "🟡", "halt": "🔴", "close": "🆘"}

    color = COLOR.get(state_val, "#64748b")
    bg    = BG.get(state_val, "rgba(100,116,139,0.1)")
    emoji = EMOJI.get(state_val, "⚪")

    return html.Div(children=[
        html.Div(children=[
            html.Span(f"{emoji} Circuit Breaker",
                      style={"font-size": "11px", "color": "#64748b", "text-transform": "uppercase",
                             "font-weight": "600", "letter-spacing": "0.5px", "margin-right": "12px"}),
            html.Span(state_val.upper(), style={"font-size": "14px", "font-weight": "700", "color": color}),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "4px"}),
        html.Div(thresholds, style={"font-size": "10px", "color": "#64748b",
                                     "font-family": "'Roboto Mono', monospace"}),
    ], style={"background-color": bg, "border": f"1px solid {color}",
              "border-radius": "6px", "padding": "12px 14px"})


def _calendar_strip():
    try:
        from data.calendar import get_calendar_context
        ctx = get_calendar_context()
    except Exception:
        return html.Div()

    dte   = ctx.get("days_to_expiry", 0)
    level = ctx.get("risk_level", "normal")
    COLOR = {"normal": "#16a34a", "elevated": "#d97706", "high": "#dc2626"}
    BG    = {"normal": "rgba(22,163,74,0.1)", "elevated": "rgba(217,119,6,0.1)", "high": "rgba(220,38,38,0.1)"}
    color = COLOR.get(level, "#64748b")
    bg    = BG.get(level, "rgba(100,116,139,0.1)")

    dte_label = "EXPIRY TODAY" if dte == 0 else (f"{dte}d to expiry" if dte <= 5
                else ctx.get("next_expiry", {}).get("label", ""))

    monthly_badge = (
        html.Span(" MONTHLY", style={"font-size": "9px", "background-color": "#7c3aed",
                                      "color": "#fff", "border-radius": "3px",
                                      "padding": "1px 4px", "font-weight": "700", "margin-left": "6px"})
        if ctx.get("monthly_expiry_week") else html.Span()
    )

    event_chips = []
    for ev in ctx.get("economic_events_14d", []):
        d = ev.get("days_away", 99)
        chip_color = "#dc2626" if d == 0 else ("#d97706" if d <= 2 else "#64748b")
        event_chips.append(html.Span(
            f"⚡ {ev.get('type','')} in {d}d",
            style={"font-size": "10px", "color": chip_color, "font-weight": "600",
                   "background-color": "rgba(220,38,38,0.12)" if d <= 2 else "rgba(100,116,139,0.1)",
                   "border": f"1px solid {chip_color}", "border-radius": "3px",
                   "padding": "2px 6px", "margin-left": "8px"},
        ))

    notes = ctx.get("trading_notes", [])
    return html.Div(children=[
        html.Div(children=[
            html.Span("F&O Calendar", style={"font-size": "11px", "color": "#64748b",
                                             "text-transform": "uppercase", "font-weight": "600",
                                             "letter-spacing": "0.5px", "margin-right": "12px"}),
            html.Span(dte_label, style={"font-size": "14px", "font-weight": "700", "color": color}),
            monthly_badge,
            html.Span(f"  [{level.upper()}]", style={"font-size": "10px", "color": color,
                                                      "font-weight": "600", "margin-left": "8px"}),
            *event_chips,
        ], style={"display": "flex", "align-items": "center", "flex-wrap": "wrap"}),
        html.Div(notes[0] if notes else "",
                 style={"font-size": "11px", "color": "#94a3b8", "margin-top": "6px",
                        "font-style": "italic"}) if notes else html.Div(),
    ], style={"background-color": bg, "border": f"1px solid {color}",
              "border-radius": "6px", "padding": "12px 14px"})


def risk_panel_component(bot_status=None, vix=None, fii_dii=None):
    if bot_status is None:
        bot_status = {}

    capital_at_risk = bot_status.get("capital_at_risk", 0) or 0
    daily_pnl       = bot_status.get("daily_pnl", 0) or 0
    open_positions  = bot_status.get("open_positions", {}) or {}
    import os
    max_capital    = int(os.getenv("MAX_CAPITAL", "1000000"))   # ₹10L default — matches .env
    max_daily_loss = max_capital * 0.03                          # 3% of actual capital

    # Capital gauge card
    capital_card = html.Div(children=[
        create_radial_gauge(capital_at_risk, max_capital, "Capital at Risk"),
        html.Div(f"₹{capital_at_risk:,.0f} / ₹{max_capital:,.0f}",
                 style={"text-align": "center", "font-family": "'Roboto Mono', monospace",
                        "font-size": "11px", "color": "#94a3b8", "margin-top": "-20px"}),
    ], style=_CARD)

    # Daily loss bar
    loss_pct = min(100, abs(daily_pnl) / max_daily_loss * 100) if daily_pnl < 0 else 0
    loss_color = "#00ff88" if daily_pnl >= 0 else ("#ffd700" if loss_pct < 80 else "#ff4444")
    daily_loss_card = html.Div(children=[
        html.Div("Daily Loss Limit", style={"font-size": "11px", "color": "#64748b",
                                             "text-transform": "uppercase", "font-weight": "600",
                                             "margin-bottom": "10px", "letter-spacing": "0.5px"}),
        html.Div(children=[html.Div(style={
            "width": f"{loss_pct}%", "height": "20px", "background-color": loss_color,
            "border-radius": "2px", "transition": "width 0.3s ease",
        })], style={"width": "100%", "height": "20px", "background-color": "#2a3547",
                    "border-radius": "2px", "margin-bottom": "8px", "overflow": "hidden"}),
        html.Div(f"₹{abs(daily_pnl):,.0f} / ₹{max_daily_loss:,.0f}  ({loss_pct:.0f}%)",
                 style={"font-family": "'Roboto Mono', monospace", "font-size": "11px", "color": loss_color}),
    ], style=_CARD)

    # Warnings
    warnings = []
    if capital_at_risk > max_capital * 0.8:
        warnings.append(html.Div("⚠️ Capital utilisation >80%", style={
            "padding": "8px", "background-color": "rgba(255,68,68,0.1)",
            "border-left": "3px solid #ff4444", "color": "#ff4444",
            "font-size": "11px", "margin-bottom": "6px", "border-radius": "2px",
        }))
    if not warnings:
        warnings = [html.Div("✅ All risk metrics within limits", style={
            "padding": "8px", "background-color": "rgba(0,255,136,0.1)",
            "border-left": "3px solid #00ff88", "color": "#00ff88",
            "font-size": "11px", "border-radius": "2px",
        })]

    warnings_card = html.Div(children=[
        html.Div("Risk Alerts", style={"font-size": "11px", "color": "#64748b",
                                        "text-transform": "uppercase", "font-weight": "600",
                                        "margin-bottom": "8px", "letter-spacing": "0.5px"}),
        *warnings,
    ], style=_CARD)

    # FII/DII
    fii_dii = fii_dii or {}
    fii_net  = fii_dii.get("fii_net")
    dii_net  = fii_dii.get("dii_net")
    flow_date= fii_dii.get("date", "")

    def _flow_chip(label, value):
        if value is None:
            return html.Span(f"{label}: N/A", style={"font-size": "11px", "color": "#64748b"})
        color = "#16a34a" if value >= 0 else "#dc2626"
        return html.Div(children=[
            html.Span(label, style={"font-size": "10px", "color": "#64748b",
                                    "text-transform": "uppercase", "font-weight": "600"}),
            html.Span(f" ₹{value:+,.0f} cr", style={"font-size": "14px", "font-weight": "700",
                                                      "color": color, "font-family": "'Roboto Mono', monospace"}),
        ], style={"display": "flex", "flex-direction": "column", "align-items": "flex-start",
                  "padding": "8px 12px", "background-color": "#1a2234",
                  "border-radius": "4px", "border": "1px solid #2a3547"})

    fii_card = html.Div(children=[
        html.Div(children=[
            html.Span("FII / DII Flows", style={"font-size": "11px", "color": "#64748b",
                                                  "text-transform": "uppercase", "font-weight": "600"}),
            html.Span(flow_date, style={"font-size": "10px", "color": "#64748b"}),
        ], style={"display": "flex", "justify-content": "space-between",
                  "align-items": "center", "margin-bottom": "10px"}),
        html.Div(children=[_flow_chip("FII", fii_net), _flow_chip("DII", dii_net)],
                 style={"display": "flex", "gap": "12px"}),
    ], style=_CARD)

    return html.Div(children=[
        # Row 1: Capital gauge + Daily loss
        html.Div(children=[
            html.Div(capital_card,    style={"flex": "1", "min-width": "220px"}),
            html.Div(daily_loss_card, style={"flex": "1", "min-width": "220px"}),
        ], style={"display": "flex", "gap": "14px", "flex-wrap": "wrap", "margin": "12px 16px"}),

        # Row 2: VIX + FII/DII
        html.Div(children=[
            html.Div(_vix_bar(vix), style={"flex": "2", "min-width": "280px"}),
            html.Div(fii_card,      style={"flex": "1", "min-width": "220px"}),
        ], style={"display": "flex", "gap": "14px", "flex-wrap": "wrap", "margin": "0 16px 14px 16px"}),

        # Row 3: Warnings
        html.Div(warnings_card, style={"margin": "0 16px 14px 16px"}),

        # Row 4: Circuit breaker
        html.Div(_circuit_breaker_strip(bot_status), style={"margin": "0 16px 14px 16px"}),

        # Row 5: Calendar
        html.Div(_calendar_strip(), style={"margin": "0 16px 14px 16px"}),
    ], style={"margin": "12px 0"})
