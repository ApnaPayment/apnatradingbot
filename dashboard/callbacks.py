"""Callbacks — All reactive dashboard callbacks."""

import io
import logging
import sys

import pandas as pd
from dash import Input, Output, State, callback, dcc, dash_table, html

sys.path.insert(0, '/Users/ankitsharma/KotaNew/algo_trader')
logger = logging.getLogger(__name__)

from dashboard import db

try:
    from data.news_fetcher import NewsFetcher as _NewsFetcher
    _news_fetcher = _NewsFetcher()
except Exception:
    _news_fetcher = None

from dashboard.components.header          import header_component
from dashboard.components.kpi_cards       import kpi_cards_component
from dashboard.components.chart           import chart_component
from dashboard.components.positions_table import positions_table_component
from dashboard.components.signal_feed     import signal_feed_component
from dashboard.components.analytics_charts import analytics_charts_component
from dashboard.components.risk_panel      import risk_panel_component

# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_TITLE = {
    "color": "#4dabf7",
    "margin": "16px",
    "font-size": "13px",
    "text-transform": "uppercase",
    "font-weight": "700",
    "letter-spacing": "0.5px",
}

_CARD = {
    "background-color": "#151a24",
    "border": "1px solid #2a3547",
    "border-radius": "6px",
    "padding": "16px",
}

_INPUT = {
    "background-color": "#1e2a3a",
    "color": "#e2e8f0",
    "border": "1px solid #2a3547",
    "border-radius": "4px",
    "padding": "6px 10px",
    "font-size": "12px",
    "font-family": "'Roboto Mono', monospace",
}

# ─────────────────────────────────────────────────────────────────────────────
# Settings page helpers (all dark-themed)
# ─────────────────────────────────────────────────────────────────────────────

def _render_optim_history():
    try:
        from ai.optimizer import load_all_optim_history
        history = load_all_optim_history(limit=6)
    except Exception:
        history = []

    if not history:
        return html.Div(
            children=[
                html.Div("Hyperparameter Optimisation History", style={**_INPUT, "border": "none",
                          "color": "#64748b", "font-size": "11px", "text-transform": "uppercase",
                          "font-weight": "600", "letter-spacing": "0.5px", "margin-bottom": "8px",
                          "padding": "0", "background-color": "transparent"}),
                html.Div("No runs yet — runs every Sunday at 00:30",
                         style={"font-size": "12px", "color": "#64748b", "font-style": "italic"}),
            ],
            style={**_CARD, "margin": "0 16px 14px 16px"},
        )

    rows = []
    for r in history:
        active = r.get("active", False)
        border_color = "#16a34a" if active else "#2a3547"
        rows.append(html.Div(children=[
            html.Div(children=[
                html.Span(r["strategy"].upper().replace("_", " "),
                          style={"font-size": "11px", "font-weight": "700", "color": "#e2e8f0"}),
                html.Span("  ● ACTIVE" if active else "  inactive",
                          style={"font-size": "9px", "color": "#16a34a" if active else "#64748b",
                                 "font-weight": "700", "margin-left": "8px"}),
                html.Span(f"  {r.get('run_at','')[:16].replace('T',' ')}",
                          style={"font-size": "10px", "color": "#64748b",
                                 "font-family": "'Roboto Mono', monospace"}),
            ], style={"display": "flex", "align-items": "center", "margin-bottom": "4px"}),
            html.Div(f"Sharpe {r['sharpe']:.2f}  |  Win {r['win_rate']:.1%}  |  AvgPnL ₹{r['total_pnl']:,.0f}",
                     style={"font-size": "11px", "color": "#94a3b8",
                            "font-family": "'Roboto Mono', monospace", "margin-bottom": "4px"}),
            html.Div("  ".join(f"{k}={v}" for k, v in r["params"].items()),
                     style={"font-size": "10px", "color": "#64748b",
                            "font-family": "'Roboto Mono', monospace"}),
        ], style={"padding": "10px 12px", "border-left": f"3px solid {border_color}",
                  "background-color": "#1a2234", "border-radius": "0 4px 4px 0",
                  "margin-bottom": "8px"}))

    return html.Div(children=[
        html.Div("Hyperparameter Optimisation History",
                 style={"color": "#64748b", "font-size": "11px", "text-transform": "uppercase",
                        "font-weight": "600", "margin-bottom": "10px", "letter-spacing": "0.5px"}),
        *rows,
    ], style={**_CARD, "margin": "0 16px 14px 16px"})


def _render_screener_results():
    try:
        from data.screener import get_latest_screener_results
        rows = get_latest_screener_results(limit=20)
    except Exception:
        rows = []

    if not rows:
        return html.Div(children=[
            html.Div("Daily Watchlist Screener", style={"color": "#64748b", "font-size": "11px",
                     "text-transform": "uppercase", "font-weight": "600", "letter-spacing": "0.5px",
                     "margin-bottom": "8px"}),
            html.Div("No run yet — runs at 08:45 daily",
                     style={"font-size": "12px", "color": "#64748b", "font-style": "italic"}),
        ], style={**_CARD, "margin": "0 16px 14px 16px"})

    run_date = rows[0].get("run_date", "")
    table_rows = []
    for r in rows:
        promoted = r.get("promoted", False)
        score = r["score"] or 0
        score_color = "#16a34a" if score >= 0.6 else ("#d97706" if score >= 0.45 else "#64748b")
        table_rows.append(html.Tr([
            html.Td(str(r["rank"]), style={"font-size": "10px", "color": "#64748b",
                                            "padding": "3px 6px", "text-align": "center"}),
            html.Td(children=[
                html.Span("● " if promoted else "  ",
                          style={"color": "#16a34a" if promoted else "#2a3547", "font-size": "9px"}),
                r["symbol"].replace("-EQ", ""),
            ], style={"font-size": "11px", "color": "#16a34a" if promoted else "#94a3b8",
                      "font-weight": "700" if promoted else "400", "padding": "3px 6px"}),
            html.Td(f"{score:.3f}", style={"font-size": "11px", "color": score_color,
                                            "font-weight": "700", "padding": "3px 6px",
                                            "font-family": "'Roboto Mono', monospace"}),
            html.Td(f"{r['momentum']:+.1f}%" if r.get("momentum") is not None else "—",
                    style={"font-size": "10px", "color": "#94a3b8", "padding": "3px 6px",
                           "font-family": "'Roboto Mono', monospace"}),
            html.Td(f"{r['volume_surge']:.1f}×" if r.get("volume_surge") is not None else "—",
                    style={"font-size": "10px", "color": "#94a3b8", "padding": "3px 6px"}),
            html.Td(f"{r['rsi']:.0f}" if r.get("rsi") is not None else "—",
                    style={"font-size": "10px", "color": "#94a3b8", "padding": "3px 6px"}),
        ]))

    return html.Div(children=[
        html.Div(children=[
            html.Span("Daily Watchlist Screener", style={"color": "#64748b", "font-size": "11px",
                      "text-transform": "uppercase", "font-weight": "600", "letter-spacing": "0.5px"}),
            html.Span(f"  {run_date}", style={"font-size": "10px", "color": "#64748b",
                                               "font-family": "'Roboto Mono', monospace"}),
        ], style={"margin-bottom": "10px"}),
        html.Table(children=[
            html.Thead(html.Tr([
                html.Th(h, style={"font-size": "9px", "color": "#64748b", "padding": "4px 6px",
                                  "text-align": "left"})
                for h in ["#", "Symbol", "Score", "20d ret", "Vol×", "RSI"]
            ])),
            html.Tbody(table_rows),
        ], style={"width": "100%", "border-collapse": "collapse"}),
        html.Div("● = promoted to active watchlist",
                 style={"font-size": "10px", "color": "#64748b", "font-style": "italic",
                        "margin-top": "8px"}),
    ], style={**_CARD, "margin": "0 16px 14px 16px"})


def _render_kelly_stats():
    try:
        from core.kelly_sizer import get_kelly_dashboard_stats
        stats = get_kelly_dashboard_stats(days=90)
    except Exception:
        stats = {"status": "error"}

    status = stats.get("status", "unknown")
    if status == "insufficient_data":
        return html.Div(children=[
            html.Div("Kelly Criterion Sizer", style={"color": "#64748b", "font-size": "11px",
                     "text-transform": "uppercase", "font-weight": "600", "letter-spacing": "0.5px",
                     "margin-bottom": "8px"}),
            html.Div(stats.get("message", "Insufficient data"),
                     style={"font-size": "12px", "color": "#d97706", "font-style": "italic"}),
        ], style={**_CARD, "margin": "0 16px 14px 16px"})

    mult = stats.get("current_multiplier", 1.0)
    mult_color = "#16a34a" if mult >= 0.5 else ("#d97706" if mult >= 0.25 else "#dc2626")

    def row(label, value):
        return html.Div(children=[
            html.Span(label, style={"font-size": "11px", "color": "#64748b",
                                    "display": "inline-block", "width": "160px"}),
            html.Span(value, style={"font-size": "12px", "color": "#e2e8f0",
                                    "font-family": "'Roboto Mono', monospace", "font-weight": "600"}),
        ], style={"margin-bottom": "5px"})

    return html.Div(children=[
        html.Div(children=[
            html.Span("Kelly Criterion Sizer", style={"color": "#64748b", "font-size": "11px",
                      "text-transform": "uppercase", "font-weight": "600", "letter-spacing": "0.5px"}),
            html.Span(f"  ● ACTIVE  mult={mult:.2f}",
                      style={"font-size": "10px", "font-weight": "700", "color": mult_color}),
        ], style={"margin-bottom": "10px"}),
        row("Win rate (90d):",     f"{stats.get('win_rate', 0):.1%}"),
        row("Avg win / loss:",     f"₹{stats.get('avg_win',0):,.0f}  /  ₹{stats.get('avg_loss',0):,.0f}"),
        row("Payoff ratio:",       f"{stats.get('payoff',0):.2f}×"),
        row("Raw Kelly (f*):",     f"{stats.get('kelly_raw',0):.3f}"),
        row("¼-Kelly (safe):",     f"{stats.get('kelly_safe',0):.3f}"),
        row("Current multiplier:", f"{mult:.2f}"),
        row("Completed trades:",   f"{stats.get('n_trades',0)}  (last {stats.get('days',90)}d)"),
    ], style={**_CARD, "margin": "0 16px 14px 16px"})


def _render_ml_status():
    try:
        from ai.ml_ensemble import get_model_stats, get_feature_importance
        stats = get_model_stats()
        fi    = get_feature_importance()
    except Exception:
        stats = {"status": "error"}
        fi    = None

    status = stats.get("status", "unknown")
    status_color = {"ready": "#16a34a", "untrained": "#d97706", "error": "#dc2626"}.get(status, "#64748b")

    if status == "ready":
        auc_str = f"  |  AUC {stats['auc']:.3f}" if stats.get("auc") else ""
        header_detail = (f"Trained on {stats['n_samples']} samples{auc_str}  "
                         f"({stats.get('trained_at','')[:16].replace('T',' ')})")
        fi_items = []
        if fi:
            pairs = sorted(zip(fi["features"], fi["importances"]), key=lambda x: x[1], reverse=True)[:5]
            for fname, imp in pairs:
                fi_items.append(html.Div(children=[
                    html.Span(fname, style={"font-size": "11px", "color": "#94a3b8",
                                            "font-family": "'Roboto Mono', monospace",
                                            "display": "inline-block", "width": "110px"}),
                    html.Div(style={"display": "inline-block", "width": f"{int(imp*200)}px",
                                    "height": "8px", "background-color": "#2563eb",
                                    "border-radius": "2px", "vertical-align": "middle",
                                    "margin-right": "8px"}),
                    html.Span(f"{imp:.3f}", style={"font-size": "10px", "color": "#64748b"}),
                ], style={"margin-bottom": "4px"}))
        body = html.Div([
            html.Div(header_detail, style={"font-size": "11px", "color": "#94a3b8",
                                            "margin-bottom": "10px",
                                            "font-family": "'Roboto Mono', monospace"}),
            html.Div("Feature Importances", style={"font-size": "10px", "color": "#64748b",
                     "text-transform": "uppercase", "font-weight": "600",
                     "margin-bottom": "6px", "letter-spacing": "0.5px"}),
            *fi_items,
        ])
    else:
        body = html.Div(stats.get("reason", "No model available"),
                        style={"font-size": "12px", "color": "#64748b", "font-style": "italic"})

    return html.Div(children=[
        html.Div(children=[
            html.Span("ML Ensemble Model", style={"color": "#64748b", "font-size": "11px",
                      "text-transform": "uppercase", "font-weight": "600", "letter-spacing": "0.5px"}),
            html.Span(f"  ● {status.upper()}",
                      style={"font-size": "10px", "font-weight": "700", "color": status_color}),
        ], style={"margin-bottom": "10px"}),
        body,
    ], style={**_CARD, "margin": "0 16px 14px 16px"})


def _render_ai_feedback():
    try:
        from ai.feedback_loop import get_ai_accuracy_stats, get_recent_decisions, get_weekly_reviews
        stats   = get_ai_accuracy_stats(days=30)
        recents = get_recent_decisions(limit=10)
        reviews = get_weekly_reviews(limit=2)
    except Exception:
        stats, recents, reviews = {}, [], []

    if stats.get("total_decided", 0) == 0:
        stats_block = html.Div("No completed AI decisions yet",
                                style={"color": "#64748b", "font-size": "12px",
                                       "font-style": "italic", "margin-bottom": "12px"})
    else:
        acc = stats["accuracy"]
        acc_color = "#16a34a" if acc >= 0.6 else ("#d97706" if acc >= 0.45 else "#dc2626")
        stats_block = html.Div(children=[
            html.Span("30-day accuracy: ", style={"color": "#64748b", "font-size": "12px"}),
            html.Span(f"{acc:.0%}", style={"color": acc_color, "font-size": "14px",
                                           "font-weight": "700", "margin-right": "16px"}),
            html.Span(f"{stats['total_decided']} decisions  |  {stats['approvals']} approved  |  "
                      f"{stats['vetoes']} vetoed  |  avg P&L ₹{stats.get('avg_pnl_on_approved',0):,.0f}",
                      style={"color": "#94a3b8", "font-size": "11px",
                             "font-family": "'Roboto Mono', monospace"}),
        ], style={"margin-bottom": "12px"})

    decision_rows = []
    for d in recents:
        pnl = d.get("outcome_pnl")
        pnl_str   = f"₹{pnl:+,.0f}" if pnl is not None else "pending"
        pnl_color = "#16a34a" if (pnl or 0) > 0 else ("#dc2626" if (pnl or 0) < 0 else "#64748b")
        correct   = {1: "✓", 0: "✗", None: "…"}.get(d.get("outcome_correct"), "…")
        approved  = d["approved"]
        decision_rows.append(html.Tr([
            html.Td(d["decided_at"][:16].replace("T", " "),
                    style={"font-size": "10px", "color": "#64748b", "padding": "3px 6px",
                           "font-family": "'Roboto Mono', monospace"}),
            html.Td(d["symbol"], style={"font-size": "11px", "font-weight": "600",
                                        "padding": "3px 6px", "color": "#e2e8f0"}),
            html.Td(d["action"], style={"font-size": "11px", "padding": "3px 6px",
                                        "color": "#94a3b8"}),
            html.Td("✅ APPROVED" if approved else "🚫 VETOED",
                    style={"font-size": "10px", "color": "#16a34a" if approved else "#dc2626",
                           "padding": "3px 6px", "font-weight": "600"}),
            html.Td(pnl_str, style={"font-size": "11px", "color": pnl_color, "padding": "3px 6px",
                                     "font-family": "'Roboto Mono', monospace"}),
            html.Td(correct, style={"font-size": "13px", "padding": "3px 6px", "text-align": "center"}),
        ]))

    decision_table = (
        html.Table(children=[
            html.Thead(html.Tr([
                html.Th(h, style={"font-size": "9px", "color": "#64748b",
                                  "padding": "4px 6px", "text-align": "left"})
                for h in ["Time", "Symbol", "Action", "Decision", "P&L", "✓"]
            ])),
            html.Tbody(decision_rows),
        ], style={"width": "100%", "border-collapse": "collapse", "margin-bottom": "14px"})
        if decision_rows else
        html.Div("No decisions recorded yet",
                 style={"color": "#64748b", "font-size": "12px", "font-style": "italic",
                        "margin-bottom": "12px"})
    )

    review_blocks = []
    for rv in reviews:
        review_blocks.append(html.Div(children=[
            html.Div(f"Week of {rv['week_start']}  —  {rv['decisions']} decisions, "
                     f"{rv.get('accuracy',0):.0%} accuracy",
                     style={"font-size": "10px", "color": "#64748b",
                            "font-family": "'Roboto Mono', monospace", "margin-bottom": "6px"}),
            html.Div(rv.get("review_text", ""),
                     style={"font-size": "12px", "color": "#e2e8f0",
                            "white-space": "pre-wrap", "line-height": "1.6"}),
        ], style={"padding": "10px 12px", "border-left": "3px solid #4dabf7",
                  "background-color": "#1a2234", "border-radius": "0 4px 4px 0",
                  "margin-bottom": "10px"}))

    if not review_blocks:
        review_blocks = [html.Div("No weekly reviews yet — first runs every Sunday at 01:00",
                                   style={"color": "#64748b", "font-size": "12px",
                                          "font-style": "italic"})]

    return html.Div(children=[
        html.Div("AI Decision Feedback Loop",
                 style={"color": "#64748b", "font-size": "11px", "text-transform": "uppercase",
                        "font-weight": "600", "margin-bottom": "10px", "letter-spacing": "0.5px"}),
        stats_block,
        decision_table,
        html.Div("Weekly Self-Review",
                 style={"color": "#64748b", "font-size": "10px", "text-transform": "uppercase",
                        "font-weight": "600", "margin-bottom": "8px", "letter-spacing": "0.5px"}),
        *review_blocks,
    ], style={**_CARD, "margin": "0 16px 14px 16px"})


def _render_settings_page():
    try:
        db_stats = db.db_stats()
    except Exception:
        db_stats = {}
    items = [
        ("Scrip Instruments", f"{db_stats.get('scrip_instruments', 0):,}"),
        ("OHLCV Candles",     f"{db_stats.get('ohlcv_candles', 0):,}"),
        ("Tick Data",         f"{db_stats.get('ticks_stored', 0):,}"),
        ("NSE Trades",        str(db_stats.get('trades_recorded', 0))),
        ("LIQ Trades",        str(db_stats.get('liq_trades', 0))),
        ("Events Logged",     str(db_stats.get('events_logged', 0))),
    ]
    rows = []
    for label, val in items:
        rows.append(html.Div(children=[
            html.Span(label, style={"font-size": "12px", "color": "#64748b",
                                    "display": "inline-block", "width": "180px"}),
            html.Span(val, style={"font-size": "12px", "color": "#e2e8f0",
                                   "font-family": "'Roboto Mono', monospace", "font-weight": "600"}),
        ], style={"margin-bottom": "6px"}))

    return html.Div(children=[
        html.Div("Database Stats", style={"color": "#64748b", "font-size": "11px",
                  "text-transform": "uppercase", "font-weight": "600",
                  "margin-bottom": "10px", "letter-spacing": "0.5px"}),
        *rows,
    ], style={**_CARD, "margin": "12px 16px 14px 16px"})


# ─────────────────────────────────────────────────────────────────────────────
# Trades filter helper
# ─────────────────────────────────────────────────────────────────────────────

def _trades_filter_bar(trades_df):
    """Build strategy & exit-reason options from actual trade data."""
    strategies   = []
    exit_reasons = []
    if trades_df is not None and not trades_df.empty:
        strategies   = sorted(trades_df["strategy"].dropna().unique().tolist())
        exit_reasons = sorted(trades_df["exit_reason"].dropna().unique().tolist())

    return html.Div(children=[
        dcc.Input(
            id="filter-symbol",
            placeholder="Search symbol…",
            debounce=True,
            style={**_INPUT, "width": "160px"},
        ),
        dcc.Dropdown(
            id="filter-strategy",
            options=[{"label": s, "value": s} for s in strategies],
            placeholder="Strategy",
            clearable=True,
            style={"width": "160px", "background-color": "#1e2a3a",
                   "color": "#111", "border": "1px solid #2a3547", "border-radius": "4px"},
        ),
        dcc.Dropdown(
            id="filter-exit-reason",
            options=[{"label": r, "value": r} for r in exit_reasons],
            placeholder="Exit reason",
            clearable=True,
            style={"width": "160px", "background-color": "#1e2a3a",
                   "color": "#111", "border": "1px solid #2a3547", "border-radius": "4px"},
        ),
        html.Button(
            "⬇ Export CSV",
            id="trades-export-btn",
            style={
                "background-color": "#1e2a3a",
                "color": "#4dabf7",
                "border": "1px solid #4dabf7",
                "border-radius": "4px",
                "padding": "6px 14px",
                "font-size": "12px",
                "cursor": "pointer",
                "font-family": "'Roboto Mono', monospace",
            },
        ),
    ], style={"display": "flex", "gap": "10px", "align-items": "center",
              "flex-wrap": "wrap", "margin": "12px 16px"})


def _build_trades_table(trades_df):
    if trades_df is None or trades_df.empty:
        return html.Div("No trades match the current filters",
                        style={"color": "#64748b", "padding": "24px 16px",
                               "font-size": "13px", "text-align": "center"})

    total_pnl = trades_df["pnl"].sum()
    wins      = len(trades_df[trades_df["pnl"] > 0])
    win_rate  = wins / len(trades_df) if len(trades_df) > 0 else 0
    pnl_color = "#00ff88" if total_pnl >= 0 else "#ff4444"

    summary = html.Div(children=[
        html.Span(f"Showing {len(trades_df)} trades",
                  style={"font-size": "12px", "color": "#64748b"}),
        html.Span(f"  |  P&L: ",
                  style={"font-size": "12px", "color": "#64748b"}),
        html.Span(f"₹{total_pnl:+,.0f}",
                  style={"font-size": "12px", "color": pnl_color, "font-weight": "700",
                         "font-family": "'Roboto Mono', monospace"}),
        html.Span(f"  |  Win rate: {win_rate:.0%}",
                  style={"font-size": "12px", "color": "#64748b"}),
    ], style={"margin": "0 16px 10px 16px"})

    table = dash_table.DataTable(
        data=trades_df.to_dict("records"),
        columns=[
            {"name": "Symbol",      "id": "symbol"},
            {"name": "Action",      "id": "action"},
            {"name": "Strategy",    "id": "strategy"},
            {"name": "Entry ₹",     "id": "entry_price",  "type": "numeric", "format": {"specifier": ",.2f"}},
            {"name": "Exit ₹",      "id": "exit_price",   "type": "numeric", "format": {"specifier": ",.2f"}},
            {"name": "Qty",         "id": "quantity"},
            {"name": "P&L ₹",       "id": "pnl",          "type": "numeric", "format": {"specifier": "+,.0f"}},
            {"name": "Exit Reason", "id": "exit_reason"},
            {"name": "Entry Time",  "id": "entry_time"},
            {"name": "Exit Time",   "id": "exit_time"},
        ],
        sort_action="native",
        page_size=20,
        page_action="native",
        style_table={"overflowX": "auto", "margin": "0 16px"},
        style_header={
            "backgroundColor": "#1e2a3a",
            "color": "#4dabf7",
            "fontWeight": "600",
            "fontSize": "11px",
            "textTransform": "uppercase",
            "border": "1px solid #2a3547",
            "letterSpacing": "0.5px",
        },
        style_cell={
            "backgroundColor": "#151a24",
            "color": "#e2e8f0",
            "fontSize": "12px",
            "padding": "8px 12px",
            "border": "1px solid #2a3547",
            "fontFamily": "'Roboto Mono', monospace",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{pnl} > 0",           "column_id": "pnl"},    "color": "#00ff88", "fontWeight": "600"},
            {"if": {"filter_query": "{pnl} < 0",           "column_id": "pnl"},    "color": "#ff4444", "fontWeight": "600"},
            {"if": {"filter_query": "{action} = 'BUY'",    "column_id": "action"}, "color": "#00ff88"},
            {"if": {"filter_query": "{action} = 'SELL'",   "column_id": "action"}, "color": "#ff4444"},
        ],
    )

    return html.Div([summary, table])


# ─────────────────────────────────────────────────────────────────────────────
# Main page routing callback
# ─────────────────────────────────────────────────────────────────────────────

from dash import ctx as _dash_ctx


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("interval-component", "n_intervals"),
)
def display_page(pathname, n_intervals):
    # Charts page: only rebuild on navigation, NOT on 3s interval tick.
    # The chart itself is refreshed by update_chart() when the dropdown changes.
    # This prevents the dropdown from resetting to the default symbol on every refresh.
    if pathname == "/charts" and _dash_ctx.triggered_id == "interval-component":
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    try:
        bot_status = db.get_bot_status()
        events_df  = db.get_events(limit=50)
        # Only fetch full trades table on pages that need it — not every 3s refresh
        trades_df    = db.get_trades(limit=2000) if pathname in ("/trades", "/analytics") else db.get_trades(limit=20)
        decisions_df = db.get_ai_decisions(limit=200)
    except Exception as e:
        logger.error(f"DB error in display_page: {e}")
        return html.Div(f"⚠ Database error: {str(e)[:120]}",
                        style={"color": "#ff4444", "padding": "24px"})

    # ── Command Center ────────────────────────────────────────────────────────
    if pathname in ("/", ""):
        try:
            ohlcv = db.get_ohlcv("NIFTYBEES-EQ", limit=200)
            return html.Div(children=[
                header_component(bot_status),
                kpi_cards_component(bot_status, trades_df),
                html.Div(children=[
                    html.Div(
                        dcc.Loading(type="default", color="#4dabf7",
                                    children=[chart_component(ohlcv, trades_df, "NIFTYBEES-EQ", decisions_df)]),
                        style={"flex": "2", "min-width": "400px"},
                    ),
                    html.Div(
                        dcc.Loading(type="default", color="#4dabf7",
                                    children=[signal_feed_component(events_df)]),
                        style={"flex": "1", "min-width": "300px"},
                    ),
                ], style={"display": "flex", "gap": "0", "flex-wrap": "wrap", "padding": "0"}),
                dcc.Loading(type="default", color="#4dabf7",
                            children=[positions_table_component(bot_status)]),
            ], style={"padding-bottom": "20px"})
        except Exception as e:
            logger.error(f"Command center error: {e}")
            return html.Div(f"⚠ Command center error: {str(e)[:120]}",
                            style={"color": "#ff4444", "padding": "24px"})

    # ── Charts ────────────────────────────────────────────────────────────────
    elif pathname == "/charts":
        try:
            symbols = db.get_symbol_list()   # full list, no cap
            symbol  = symbols[0] if symbols else "NIFTYBEES-EQ"
            ohlcv   = db.get_ohlcv(symbol, limit=200)
            return html.Div(children=[
                header_component(bot_status),
                html.Div(children=[
                    html.Label("Symbol:", style={"color": "#64748b", "font-size": "12px",
                                                  "margin-right": "8px"}),
                    dcc.Dropdown(
                        id="chart-symbol-selector",
                        options=[{"label": s, "value": s} for s in symbols],
                        value=symbol,
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        style={"width": "220px", "background-color": "#1e2a3a",
                               "color": "#111", "border": "1px solid #2a3547",
                               "border-radius": "4px"},
                    ),
                ], style={"margin": "12px 16px", "display": "flex", "gap": "10px",
                           "align-items": "center"}),
                # No dcc.Loading wrapper — avoids spinner flash on every auto-refresh
                html.Div(id="chart-container",
                         children=[chart_component(ohlcv, trades_df, symbol, decisions_df)]),
                # Separate slow interval for chart-only refresh (60s — candles build every 5min)
                dcc.Interval(id="chart-interval", interval=60_000, n_intervals=0),
            ], style={"padding-bottom": "20px"})
        except Exception as e:
            logger.error(f"Charts page error: {e}")
            return html.Div(f"⚠ Charts error: {str(e)[:120]}",
                            style={"color": "#ff4444", "padding": "24px"})

    # ── Trades ────────────────────────────────────────────────────────────────
    elif pathname == "/trades":
        try:
            # NSE Bot trades
            nse_section = html.Div([
                html.H3("NSE Bot Trades", style={"color": "#4dabf7", "margin": "12px 0 8px 0", "fontSize": "16px"}),
                _trades_filter_bar(trades_df),
                dcc.Loading(
                    id="loading-trades-table", type="default", color="#4dabf7",
                    children=[html.Div(id="trades-table-content",
                                       children=_build_trades_table(trades_df))],
                ),
            ])

            # Liquidity Engine trades (MCX + BSE paper)
            liq_trades_df  = db.get_liq_trades(limit=500)
            liq_stats      = db.get_liq_today_stats()
            liq_status     = db.get_liq_bot_status()
            liq_decisions  = db.get_liq_decisions(limit=50)

            if not liq_trades_df.empty:
                liq_display_cols = ["symbol", "exchange", "action", "strategy",
                                    "entry_price", "exit_price", "quantity",
                                    "pnl", "exit_reason", "entry_time"]
                liq_display_cols = [c for c in liq_display_cols if c in liq_trades_df.columns]
                liq_table = dash_table.DataTable(
                    data=liq_trades_df[liq_display_cols].to_dict("records"),
                    columns=[{"name": c.replace("_", " ").title(), "id": c} for c in liq_display_cols],
                    sort_action="native",
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#1e2430", "color": "#4dabf7",
                                  "fontWeight": "600", "border": "1px solid #2a3142"},
                    style_cell={"backgroundColor": "#151a24", "color": "#e0e0e0",
                                "border": "1px solid #2a3142", "padding": "6px 10px",
                                "fontFamily": "Roboto Mono, monospace", "fontSize": "12px"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{pnl} > 0"},
                         "color": "#00ff88", "fontWeight": "600"},
                        {"if": {"filter_query": "{pnl} < 0"},
                         "color": "#ff4444"},
                    ],
                )
            else:
                liq_table = html.Div("No liquidity engine trades yet",
                                     style={"color": "#888", "padding": "16px"})

            # ── Liquidity signal log ──────────────────────────────────────────
            if not liq_decisions.empty:
                log_rows = []
                for _, row in liq_decisions.iterrows():
                    approved  = bool(row.get("approved", 0))
                    icon      = "✅" if approved else "🚫"
                    color     = "#00ff88" if approved else "#ff6666"
                    symbol    = str(row.get("symbol", ""))
                    action    = str(row.get("action", ""))
                    price     = row.get("signal_price", 0) or 0
                    ts        = str(row.get("decided_at", ""))[:16]
                    reason    = str(row.get("veto_reason") or row.get("ai_reasoning") or "")[:80]
                    outcome   = str(row.get("outcome") or "open")
                    pnl_val   = row.get("outcome_pnl")
                    pnl_str   = f" → ₹{pnl_val:+.2f}" if pnl_val is not None else ""
                    pnl_color = "#00ff88" if (pnl_val or 0) > 0 else "#ff4444" if (pnl_val or 0) < 0 else "#aaa"
                    log_rows.append(html.Div([
                        html.Span(f"{icon} {ts}  ", style={"color": "#888", "fontSize": "11px"}),
                        html.Span(f"{action} {symbol} @ ₹{price:.0f}  ", style={"color": color, "fontWeight": "600"}),
                        html.Span(f"{outcome}{pnl_str}  ", style={"color": pnl_color}),
                        html.Span(reason, style={"color": "#666", "fontSize": "11px"}),
                    ], style={"padding": "4px 8px", "borderBottom": "1px solid #1e2430",
                               "fontFamily": "Roboto Mono, monospace", "fontSize": "12px"}))
                liq_signal_log = html.Div(log_rows,
                    style={"backgroundColor": "#0d0f14", "border": "1px solid #2a3142",
                           "borderRadius": "6px", "maxHeight": "300px", "overflowY": "auto",
                           "marginBottom": "12px"})
            else:
                liq_signal_log = html.Div("No signals yet today",
                    style={"color": "#888", "padding": "10px", "fontSize": "12px"})

            # ── Open MCX positions ────────────────────────────────────────────
            open_pos = liq_status.get("open_positions", {})
            if open_pos:
                pos_rows = []
                for sym, pos in open_pos.items():
                    entry  = pos.get("entry_price", 0)
                    sl     = pos.get("stop_loss", 0)
                    tgt    = pos.get("target", 0)
                    qty    = pos.get("quantity", 1)
                    act    = pos.get("action", "BUY")
                    et     = str(pos.get("entry_time", ""))[:16]
                    sl_dist = abs(entry - sl) / entry * 100 if entry else 0
                    pos_rows.append(html.Tr([
                        html.Td(sym, style={"color": "#ffd700"}),
                        html.Td(act, style={"color": "#00ff88" if act == "BUY" else "#ff6666"}),
                        html.Td(f"₹{entry:.2f}"),
                        html.Td(f"₹{sl:.2f}", style={"color": "#ff4444"}),
                        html.Td(f"₹{tgt:.2f}", style={"color": "#00ff88"}),
                        html.Td(str(qty)),
                        html.Td(f"{sl_dist:.1f}% to SL"),
                        html.Td(et, style={"color": "#888", "fontSize": "11px"}),
                    ], style={"borderBottom": "1px solid #1e2430"}))
                _th = {"backgroundColor": "#1e2430", "color": "#4dabf7",
                       "padding": "6px 10px", "fontFamily": "Roboto Mono, monospace", "fontSize": "12px"}
                _td_style = {"backgroundColor": "#151a24", "color": "#e0e0e0",
                             "padding": "5px 10px", "fontFamily": "Roboto Mono, monospace", "fontSize": "12px"}
                open_pos_table = html.Table([
                    html.Thead(html.Tr([html.Th(h, style=_th) for h in
                        ["Symbol","Action","Entry","Stop Loss","Target","Qty","SL Distance","Entry Time"]])),
                    html.Tbody(pos_rows, style=_td_style),
                ], style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "12px"})
            else:
                open_pos_table = html.Div("No open MCX positions",
                    style={"color": "#888", "padding": "10px", "fontSize": "12px"})

            segments = ", ".join(liq_status.get("segments_active", [])) or "—"
            liq_section = html.Div([
                html.H3("Liquidity Engine — MCX & BSE Paper Trades",
                        style={"color": "#ffd700", "margin": "24px 0 8px 0", "fontSize": "16px"}),
                html.Div([
                    html.Span(f"Status: {liq_status.get('status', '—').upper()}  ",
                              style={"color": "#00ff88" if liq_status.get("status") == "running" else "#888"}),
                    html.Span(f"Segments: {segments}  ",
                              style={"color": "#aaa", "marginLeft": "16px"}),
                    html.Span(f"Today: {liq_stats['trades']} trades | P&L ₹{liq_stats['total_pnl']:+.2f} | "
                              f"MCX: {liq_stats['mcx_trades']} | BSE: {liq_stats['bse_trades']}",
                              style={"color": "#aaa", "marginLeft": "16px"}),
                ], style={"marginBottom": "8px", "fontSize": "13px"}),
                html.Div("📋 Signal Log (latest 50)", style={"color": "#4dabf7", "fontSize": "13px",
                                                              "fontWeight": "600", "marginBottom": "4px"}),
                liq_signal_log,
                html.Div("📌 Open Positions", style={"color": "#4dabf7", "fontSize": "13px",
                                                      "fontWeight": "600", "margin": "12px 0 4px 0"}),
                open_pos_table,
                html.Div("📊 Completed Trades", style={"color": "#4dabf7", "fontSize": "13px",
                                                        "fontWeight": "600", "margin": "12px 0 4px 0"}),
                liq_table,
            ])

            return html.Div(children=[
                header_component(bot_status),
                html.Div("Trade History", style=_SECTION_TITLE),
                nse_section,
                liq_section,
            ], style={"padding-bottom": "20px"})
        except Exception as e:
            logger.error(f"Trades page error: {e}")
            return html.Div(f"⚠ Trades error: {str(e)[:120]}",
                            style={"color": "#ff4444", "padding": "24px"})

    # ── Analytics ─────────────────────────────────────────────────────────────
    elif pathname == "/analytics":
        try:
            return html.Div(children=[
                header_component(bot_status),
                html.Div("Performance Analytics", style=_SECTION_TITLE),
                dcc.Loading(type="default", color="#4dabf7",
                            children=[analytics_charts_component(trades_df, bot_status=bot_status)]),
            ], style={"padding-bottom": "20px"})
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return html.Div(f"⚠ Analytics error: {str(e)[:120]}",
                            style={"color": "#ff4444", "padding": "24px"})

    # ── Risk ──────────────────────────────────────────────────────────────────
    elif pathname == "/risk":
        try:
            vix     = _news_fetcher.get_india_vix()    if _news_fetcher else None
            fii_dii = _news_fetcher.get_fii_dii_flows() if _news_fetcher else {}
            return html.Div(children=[
                header_component(bot_status),
                html.Div("Risk Monitor", style=_SECTION_TITLE),
                dcc.Loading(type="default", color="#4dabf7",
                            children=[risk_panel_component(bot_status, vix=vix, fii_dii=fii_dii)]),
            ], style={"padding-bottom": "20px"})
        except Exception as e:
            logger.error(f"Risk error: {e}")
            return html.Div(f"⚠ Risk error: {str(e)[:120]}",
                            style={"color": "#ff4444", "padding": "24px"})

    # ── Settings ──────────────────────────────────────────────────────────────
    elif pathname == "/settings":
        try:
            return html.Div(children=[
                header_component(bot_status),
                html.Div("Settings & Info", style=_SECTION_TITLE),
                dcc.Loading(type="default", color="#4dabf7", children=[_render_settings_page()]),
                dcc.Loading(type="default", color="#4dabf7", children=[_render_optim_history()]),
                dcc.Loading(type="default", color="#4dabf7", children=[_render_screener_results()]),
                dcc.Loading(type="default", color="#4dabf7", children=[_render_kelly_stats()]),
                dcc.Loading(type="default", color="#4dabf7", children=[_render_ml_status()]),
                dcc.Loading(type="default", color="#4dabf7", children=[_render_ai_feedback()]),
            ], style={"padding-bottom": "20px"})
        except Exception as e:
            logger.error(f"Settings error: {e}")
            return html.Div(f"⚠ Settings error: {str(e)[:120]}",
                            style={"color": "#ff4444", "padding": "24px"})

    return html.Div(children=[
        header_component(bot_status),
        html.Div("404 — Page not found",
                 style={"padding": "40px", "text-align": "center", "color": "#64748b"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Chart symbol selector callback
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("chart-container", "children"),
    Input("chart-symbol-selector", "value"),
    Input("chart-interval", "n_intervals"),
    prevent_initial_call=True,
)
def update_chart(symbol, n_intervals):
    if not symbol:
        return []
    try:
        trades_df    = db.get_trades(limit=2000)
        decisions_df = db.get_ai_decisions(limit=200)
        ohlcv        = db.get_ohlcv(symbol, limit=200)
        return chart_component(ohlcv, trades_df, symbol, decisions_df)
    except Exception as e:
        return html.Div(f"⚠ Chart error: {e}", style={"color": "#ff4444", "padding": "16px"})


# ─────────────────────────────────────────────────────────────────────────────
# Trades filter callback
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("trades-table-content", "children"),
    Input("filter-symbol",      "value"),
    Input("filter-strategy",    "value"),
    Input("filter-exit-reason", "value"),
    prevent_initial_call=True,
)
def filter_trades(symbol, strategy, exit_reason):
    try:
        df = db.get_trades(limit=2000)
        if df.empty:
            return _build_trades_table(df)
        if symbol:
            df = df[df["symbol"].str.contains(symbol.upper(), na=False)]
        if strategy:
            df = df[df["strategy"] == strategy]
        if exit_reason:
            df = df[df["exit_reason"] == exit_reason]
        return _build_trades_table(df)
    except Exception as e:
        return html.Div(f"Filter error: {e}", style={"color": "#ff4444", "padding": "16px"})


# ─────────────────────────────────────────────────────────────────────────────
# Trades CSV export callback
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("trades-download", "data"),
    Input("trades-export-btn", "n_clicks"),
    State("filter-symbol",      "value"),
    State("filter-strategy",    "value"),
    State("filter-exit-reason", "value"),
    prevent_initial_call=True,
)
def export_trades_csv(n_clicks, symbol, strategy, exit_reason):
    if not n_clicks:
        return None
    try:
        df = db.get_trades(limit=2000)
        if not df.empty:
            if symbol:
                df = df[df["symbol"].str.contains(symbol.upper(), na=False)]
            if strategy:
                df = df[df["strategy"] == strategy]
            if exit_reason:
                df = df[df["exit_reason"] == exit_reason]
        return dcc.send_data_frame(df.to_csv, "algotrader_trades.csv", index=False)
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        return None
