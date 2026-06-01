"""Analytics charts — Equity curve, monthly P&L, strategy bars, exit donut,
sector pie, correlation heatmap, performance KPIs, win-rate histogram."""

from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#151a24",
    plot_bgcolor="#0d0f14",
    font=dict(family="'Roboto Mono', monospace", size=11, color="#94a3b8"),
    margin={"l": 50, "r": 30, "t": 40, "b": 40},
    height=300,
)


def _empty_fig(msg="No trades yet"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color="#64748b", size=12))
    fig.update_layout(**_DARK)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ─── individual charts ────────────────────────────────────────────────────────

def equity_curve_chart(trades_df):
    if trades_df is None or trades_df.empty:
        return _empty_fig()
    df = trades_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("exit_time")
    df["cumulative_pnl"] = df["pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["exit_time"], y=df["cumulative_pnl"],
        fill="tozeroy", name="Equity Curve",
        line=dict(color="#00ff88", width=2),
        fillcolor="rgba(0,255,136,0.15)",
        hovertemplate="<b>P&L:</b> ₹%{y:,.0f}<br><b>Date:</b> %{x|%d %b %Y}",
    ))
    fig.update_layout(**_DARK, title="Equity Curve — Cumulative P&L",
                      xaxis_title="Date", yaxis_title="Cumulative P&L (₹)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def monthly_pnl_chart(trades_df):
    if trades_df is None or trades_df.empty:
        return _empty_fig()
    df = trades_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["month"] = df["exit_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    monthly.index = monthly.index.strftime("%b %Y")
    colors = ["#00ff88" if x >= 0 else "#ff4444" for x in monthly.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly.values,
        marker=dict(color=colors),
        text=[f"₹{x:,.0f}" for x in monthly.values],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>P&L: ₹%{y:,.0f}",
        showlegend=False,
    ))
    fig.update_layout(**_DARK, title="Monthly P&L", xaxis_title="Month", yaxis_title="P&L (₹)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def strategy_comparison_chart(trades_df):
    if trades_df is None or trades_df.empty:
        return _empty_fig()
    df = trades_df.copy()
    sp = df.groupby("strategy")["pnl"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
    colors = ["#00ff88" if x >= 0 else "#ff4444" for x in sp["sum"].values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sp.index, y=sp["sum"],
        marker=dict(color=colors),
        text=[f"₹{x:,.0f}" for x in sp["sum"].values],
        textposition="auto",
        customdata=np.column_stack((sp["count"], sp["mean"])),
        hovertemplate="<b>%{x}</b><br>Total: ₹%{y:,.0f}<br>Trades: %{customdata[0]}<br>Avg: ₹%{customdata[1]:,.0f}",
        showlegend=False,
    ))
    fig.update_layout(**_DARK, title="Strategy Comparison",
                      xaxis_title="Strategy", yaxis_title="Total P&L (₹)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def exit_reason_chart(trades_df):
    if trades_df is None or trades_df.empty:
        return _empty_fig()
    df = trades_df.copy()
    counts = df["exit_reason"].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.4,
        marker=dict(colors=["#00ff88", "#ff4444", "#ffd700", "#4dabf7"][:len(counts)]),
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}",
    )])
    fig.update_layout(**_DARK, title="Exit Reason Distribution")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def win_rate_histogram(trades_df):
    if trades_df is None or trades_df.empty:
        return _empty_fig("No trades for histogram")
    df = trades_df.copy()
    if "pnl" not in df.columns:
        return _empty_fig()
    wins   = df[df["pnl"] > 0]["pnl"].values
    losses = df[df["pnl"] < 0]["pnl"].values

    fig = go.Figure()
    if len(wins) > 0:
        fig.add_trace(go.Histogram(x=wins,   name="Win",  marker_color="#00ff88", opacity=0.75, nbinsx=15))
    if len(losses) > 0:
        fig.add_trace(go.Histogram(x=losses, name="Loss", marker_color="#ff4444", opacity=0.75, nbinsx=15))
    fig.update_layout(**_DARK, title="P&L Distribution",
                      xaxis_title="P&L (₹)", yaxis_title="# Trades", barmode="overlay")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def correlation_heatmap(bot_status=None):
    open_positions = (bot_status or {}).get("open_positions", {})
    symbols = list(open_positions.keys())

    if len(symbols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need ≥2 open positions for correlation",
                           showarrow=False, font=dict(color="#64748b", size=12))
        fig.update_layout(**_DARK, title="Position Correlation")
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    try:
        import sys; sys.path.insert(0, '/Users/ankitsharma/KotaNew/algo_trader')
        from data.data_manager import DataManager
        dm = DataManager()
        closes = {}
        for sym in symbols:
            df = dm.get_ohlcv(sym, limit=60)
            if df is not None and len(df) >= 20:
                closes[sym] = df["close"].values

        if len(closes) < 2:
            raise ValueError("Insufficient OHLCV")

        min_len = min(len(v) for v in closes.values())
        returns = {s: np.diff(np.log(v[-min_len:])) for s, v in closes.items()}
        corr = pd.DataFrame(returns).corr().round(2)
        labels = [s.replace("-EQ", "") for s in corr.columns]

        fig = go.Figure(go.Heatmap(
            z=corr.values.tolist(), x=labels, y=labels,
            colorscale=[[0, "#dc2626"], [0.5, "#1a2234"], [1, "#16a34a"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr.values.round(2).tolist(), texttemplate="%{text}",
            showscale=True, colorbar=dict(thickness=12, len=0.8),
        ))
        fig.update_layout(**{**_DARK, "height": 280}, title="Position Correlation (60-candle returns)")
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Correlation unavailable: {str(e)[:60]}",
                           showarrow=False, font=dict(color="#64748b", size=11))
        fig.update_layout(**{**_DARK, "height": 280}, title="Position Correlation")
        return dcc.Graph(figure=fig, config={"displayModeBar": False})


def sector_pie_chart(bot_status=None):
    open_positions = (bot_status or {}).get("open_positions", {})
    if not open_positions:
        fig = go.Figure()
        fig.add_annotation(text="No open positions", showarrow=False,
                           font=dict(color="#64748b", size=12))
        fig.update_layout(**{**_DARK, "height": 280}, title="Sector Allocation")
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    try:
        import sys; sys.path.insert(0, '/Users/ankitsharma/KotaNew/algo_trader')
        from data.portfolio_analytics import SECTOR_MAP
        sector_vals: dict = {}
        for sym, pos in open_positions.items():
            base   = sym.split("-")[0]
            sector = SECTOR_MAP.get(base, "Other")
            val    = float(pos.get("entry_price", 0)) * int(pos.get("quantity", 1))
            sector_vals[sector] = sector_vals.get(sector, 0) + val

        labels = list(sector_vals.keys())
        values = [sector_vals[s] for s in labels]
        colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#9333ea", "#64748b", "#0891b2"]

        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.4,
            marker=dict(colors=colors[:len(labels)]),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}",
        ))
        fig.update_layout(**{**_DARK, "height": 280}, title="Sector Allocation", showlegend=False)
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Unavailable: {str(e)[:60]}", showarrow=False,
                           font=dict(color="#64748b", size=11))
        fig.update_layout(**{**_DARK, "height": 280}, title="Sector Allocation")
        return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ─── Performance KPI strip ────────────────────────────────────────────────────

def _compute_kpis(trades_df):
    if trades_df is None or trades_df.empty:
        return {}
    df = trades_df.copy()
    pnl = df["pnl"]
    total = pnl.sum()
    wins  = pnl[pnl > 0]
    losses= pnl[pnl < 0]
    n     = len(pnl)
    win_rate = len(wins) / n if n > 0 else 0

    # Sharpe (daily, assume 252 trading days)
    if pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Profit Factor
    gross_win  = wins.sum()  if len(wins)   > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown (on cumulative P&L)
    cum = pnl.cumsum()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max)
    max_dd = drawdown.min()

    # Calmar = annualised return / max drawdown
    avg_daily = pnl.mean() * 252
    calmar = avg_daily / abs(max_dd) if max_dd < 0 else 0

    return {
        "total_pnl":     total,
        "win_rate":      win_rate,
        "sharpe":        sharpe,
        "profit_factor": profit_factor,
        "max_drawdown":  max_dd,
        "calmar":        calmar,
        "n_trades":      n,
    }


def performance_kpi_strip(trades_df):
    kpis = _compute_kpis(trades_df)

    def kpi_box(label, value, color="#e2e8f0", sub=""):
        return html.Div(
            children=[
                html.Div(label, style={"font-size": "10px", "color": "#64748b",
                                       "text-transform": "uppercase", "font-weight": "600",
                                       "letter-spacing": "0.5px", "margin-bottom": "4px"}),
                html.Div(value, style={"font-size": "20px", "font-weight": "700",
                                       "color": color, "font-family": "'Roboto Mono', monospace"}),
                html.Div(sub, style={"font-size": "10px", "color": "#64748b",
                                     "margin-top": "2px"}),
            ],
            style={"flex": "1", "min-width": "120px", "padding": "14px 16px",
                   "background-color": "#151a24", "border": "1px solid #2a3547",
                   "border-radius": "6px"},
        )

    if not kpis:
        return html.Div(
            "No trades — KPIs will appear after first completed trade",
            style={"color": "#64748b", "padding": "14px 16px", "font-size": "12px",
                   "background-color": "#151a24", "border": "1px solid #2a3547",
                   "border-radius": "6px"},
        )

    sharpe_color = "#00ff88" if kpis["sharpe"] >= 1.0 else ("#ffd700" if kpis["sharpe"] >= 0.5 else "#ff4444")
    pf = kpis["profit_factor"]
    pf_str = f"{pf:.2f}×" if pf != float("inf") else "∞"
    pf_color = "#00ff88" if pf >= 1.5 else ("#ffd700" if pf >= 1.0 else "#ff4444")

    return html.Div(
        children=[
            kpi_box("Total P&L",
                    f"₹{kpis['total_pnl']:+,.0f}",
                    color="#00ff88" if kpis["total_pnl"] >= 0 else "#ff4444"),
            kpi_box("Win Rate",
                    f"{kpis['win_rate']:.0%}",
                    color="#00ff88" if kpis["win_rate"] >= 0.55 else "#ffd700",
                    sub=f"{kpis['n_trades']} trades"),
            kpi_box("Sharpe",  f"{kpis['sharpe']:.2f}", color=sharpe_color, sub="annualised"),
            kpi_box("Profit Factor", pf_str, color=pf_color),
            kpi_box("Max Drawdown",  f"₹{kpis['max_drawdown']:,.0f}", color="#ff4444"),
            kpi_box("Calmar",  f"{kpis['calmar']:.2f}", color="#4dabf7", sub="ann/MDD"),
        ],
        style={"display": "flex", "gap": "10px", "flex-wrap": "wrap",
               "margin": "0 16px 16px 16px"},
    )


# ─── Main component ───────────────────────────────────────────────────────────

def analytics_charts_component(trades_df=None, bot_status=None):
    return html.Div(
        children=[
            # Row 0 — Performance KPI strip
            performance_kpi_strip(trades_df),

            # Row 1 — Equity curve + Monthly P&L
            html.Div(
                children=[
                    html.Div(equity_curve_chart(trades_df),    style={"flex": "1", "min-width": "300px"}),
                    html.Div(monthly_pnl_chart(trades_df),     style={"flex": "1", "min-width": "300px"}),
                ],
                style={"display": "flex", "gap": "16px", "flex-wrap": "wrap",
                       "margin": "0 16px 16px 16px"},
            ),

            # Row 2 — Strategy comparison + Exit reason
            html.Div(
                children=[
                    html.Div(strategy_comparison_chart(trades_df), style={"flex": "1", "min-width": "300px"}),
                    html.Div(exit_reason_chart(trades_df),         style={"flex": "1", "min-width": "300px"}),
                ],
                style={"display": "flex", "gap": "16px", "flex-wrap": "wrap",
                       "margin": "0 16px 16px 16px"},
            ),

            # Row 3 — Win-rate histogram + Sector pie
            html.Div(
                children=[
                    html.Div(win_rate_histogram(trades_df),    style={"flex": "2", "min-width": "300px"}),
                    html.Div(sector_pie_chart(bot_status),     style={"flex": "1", "min-width": "280px"}),
                ],
                style={"display": "flex", "gap": "16px", "flex-wrap": "wrap",
                       "margin": "0 16px 16px 16px"},
            ),

            # Row 4 — Correlation heatmap
            html.Div(
                correlation_heatmap(bot_status),
                style={"margin": "0 16px 16px 16px"},
            ),
        ],
    )
