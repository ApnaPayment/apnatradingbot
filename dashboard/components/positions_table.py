"""Positions table component — Live open positions with unrealised P&L and SL proximity."""

from dash import html, dash_table
import pandas as pd


def positions_table_component(bot_status=None, quotes=None):
    if bot_status is None:
        bot_status = {}
    if quotes is None:
        quotes = {}

    open_positions = bot_status.get("open_positions", {}) or {}

    if not open_positions:
        return html.Div(
            "No open positions",
            style={
                "padding": "20px",
                "text-align": "center",
                "color": "#64748b",
                "font-size": "13px",
                "background-color": "#151a24",
                "border": "1px solid #2a3547",
                "border-radius": "6px",
                "margin": "12px 16px",
            },
        )

    rows = []
    for symbol, pos in open_positions.items():
        quote        = quotes.get(symbol, {})
        current_price= float(quote.get("ltp", pos.get("entry_price", 0)))
        entry_price  = float(pos.get("entry_price", 0))
        stop_loss    = float(pos.get("stop_loss", 0))
        target       = float(pos.get("target", 0))
        action       = pos.get("action", "BUY")
        quantity     = int(pos.get("quantity", 0))
        strategy     = pos.get("strategy", "")

        unrealised_pnl = (
            (current_price - entry_price) * quantity if action == "BUY"
            else (entry_price - current_price) * quantity
        )

        sl_proximity = (
            ((current_price - stop_loss) / (entry_price - stop_loss) * 100)
            if entry_price != stop_loss else 0
        )
        target_proximity = (
            ((current_price - entry_price) / (target - entry_price) * 100)
            if target != entry_price else 0
        )

        if sl_proximity < 20:
            status = "🚨 Near SL"
        elif target_proximity > 80:
            status = "📈 Near Target"
        else:
            status = "📍 Hold"

        sl_indicator = ""
        if action == "BUY" and stop_loss > entry_price:
            sl_indicator = " ▲"
        elif action == "SELL" and stop_loss < entry_price:
            sl_indicator = " ▼"

        rows.append({
            "Symbol":   symbol,
            "Action":   action,
            "Entry":    f"₹{entry_price:.2f}",
            "Current":  f"₹{current_price:.2f}",
            "SL":       f"₹{stop_loss:.2f}{sl_indicator}",
            "Target":   f"₹{target:.2f}",
            "P&L":      f"₹{unrealised_pnl:+,.0f}",
            "Status":   status,
            "Strategy": strategy,
        })

    df = pd.DataFrame(rows)

    return html.Div(
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            sort_action="native",
            style_table={"overflowX": "auto", "margin": "0"},
            style_header={
                "backgroundColor": "#1e2a3a",
                "color": "#4dabf7",
                "fontWeight": "600",
                "textTransform": "uppercase",
                "fontSize": "11px",
                "borderBottom": "1px solid #2a3547",
                "letterSpacing": "0.5px",
                "border": "1px solid #2a3547",
            },
            style_cell={
                "textAlign": "left",
                "padding": "10px",
                "fontFamily": "'Roboto Mono', monospace",
                "fontSize": "12px",
                "backgroundColor": "#151a24",
                "color": "#e2e8f0",
                "border": "1px solid #2a3547",
            },
            style_data_conditional=[
                {"if": {"column_id": "P&L", "filter_query": "{P&L} contains +"}, "color": "#00ff88", "fontWeight": "600"},
                {"if": {"column_id": "P&L", "filter_query": "{P&L} contains -"}, "color": "#ff4444", "fontWeight": "600"},
                {"if": {"column_id": "Action", "filter_query": "{Action} = 'BUY'"},  "color": "#00ff88"},
                {"if": {"column_id": "Action", "filter_query": "{Action} = 'SELL'"}, "color": "#ff4444"},
                {"if": {"column_id": "Status", "filter_query": "{Status} contains 🚨"}, "color": "#ff4444"},
                {"if": {"column_id": "Status", "filter_query": "{Status} contains 📈"}, "color": "#00ff88"},
            ],
        ),
        style={"margin": "12px 16px"},
    )
