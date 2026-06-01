"""Signal feed component — Scrolling event log (signals, vetoes, executions)."""

from dash import html
import pandas as pd
import json
from datetime import datetime

EVENT_STYLES = {
    "SIGNAL_BUY":   ("🟢", "#16a34a"),
    "SIGNAL_SELL":  ("🔴", "#dc2626"),
    "EXECUTION":    ("✅", "#2563eb"),
    "VETO":         ("❌", "#d97706"),
    "TRAILING_SL":  ("📍", "#9333ea"),
    "CYCLE":        ("🔄", "#2a3547"),
}


def _parse_time(timestamp):
    try:
        dt = datetime.fromisoformat(str(timestamp))
        today = datetime.now().date()
        if dt.date() == today:
            return dt.strftime("%H:%M:%S")
        else:
            return dt.strftime("%d %b %H:%M")   # e.g. "30 May 15:28"
    except Exception:
        return "--:--:--"


def signal_feed_component(events_df=None, limit=40):
    if events_df is None or events_df.empty:
        return html.Div(
            "No recent events",
            style={
                "padding": "20px",
                "text-align": "center",
                "color": "#64748b",
                "font-size": "12px",
                "background-color": "#151a24",
                "border": "1px solid #2a3547",
                "border-radius": "6px",
                "margin": "12px 16px",
                "min-height": "80px",
            },
        )

    df = events_df.head(limit)
    items = []

    for _, row in df.iterrows():
        etype   = (row.get("event_type") or "").upper()
        emoji, border_color = EVENT_STYLES.get(etype, ("ℹ️", "#2a3547"))
        time_str = _parse_time(row.get("timestamp", ""))
        message  = row.get("message", "")
        symbol   = row.get("symbol", "")

        # Tool chips from metadata
        tool_chips = []
        try:
            meta = row.get("metadata", "{}")
            if isinstance(meta, str):
                meta = json.loads(meta)
            for t in meta.get("tools_used", []):
                short = t.replace("get_", "").replace("_", " ")
                tool_chips.append(html.Span(
                    short,
                    style={
                        "background-color": "rgba(37,99,235,0.15)",
                        "color": "#4dabf7",
                        "border-radius": "3px",
                        "font-size": "9px",
                        "padding": "1px 5px",
                        "margin-left": "4px",
                        "font-family": "'Roboto Mono', monospace",
                    },
                ))
        except Exception:
            pass

        items.append(html.Div(
            children=[
                html.Span(emoji,    style={"margin-right": "6px", "font-size": "11px"}),
                html.Span(time_str, style={"color": "#64748b", "font-family": "'Roboto Mono', monospace", "font-size": "10px", "margin-right": "8px"}),
                html.Span(message,  style={"color": "#e2e8f0", "font-size": "11px"}),
                html.Span(f" [{symbol}]" if symbol else "", style={"color": "#4dabf7", "font-size": "10px", "margin-left": "4px", "font-family": "'Roboto Mono', monospace"}),
                *tool_chips,
            ],
            style={
                "display": "flex",
                "align-items": "center",
                "flex-wrap": "wrap",
                "padding": "7px 10px",
                "border-left": f"3px solid {border_color}",
                "margin-bottom": "5px",
                "background-color": "#1a2234",
                "border-radius": "2px",
                "font-size": "11px",
            },
        ))

    return html.Div(
        children=[
            html.Div(
                "Signal Feed",
                style={"font-size": "11px", "color": "#64748b", "text-transform": "uppercase",
                       "font-weight": "600", "letter-spacing": "0.5px", "margin-bottom": "8px"},
            ),
            *items,
        ],
        style={
            "background-color": "#151a24",
            "border": "1px solid #2a3547",
            "border-radius": "6px",
            "padding": "12px",
            "max-height": "420px",
            "overflow-y": "auto",
            "margin": "12px 16px",
        },
    )
