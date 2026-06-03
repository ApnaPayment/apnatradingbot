"""Chart component — Candlestick with EMA/BB overlays, trade markers, range buttons."""

from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#151a24",
    plot_bgcolor="#0d0f14",
    font=dict(family="'Roboto Mono', monospace", size=11, color="#94a3b8"),
    xaxis=dict(gridcolor="#1e2a3a", showgrid=True, rangeslider_visible=False),
    yaxis=dict(gridcolor="#1e2a3a", showgrid=True),
    hovermode="x unified",
    margin={"l": 50, "r": 30, "t": 50, "b": 40},
    height=420,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

_RANGE_BUTTONS = dict(
    rangeselector=dict(
        buttons=[
            dict(count=1,  label="1D",  step="day",   stepmode="backward"),
            dict(count=5,  label="5D",  step="day",   stepmode="backward"),
            dict(count=1,  label="1M",  step="month", stepmode="backward"),
            dict(count=3,  label="3M",  step="month", stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor="#1e2a3a",
        activecolor="#4dabf7",
        font=dict(color="#e2e8f0", size=11),
        bordercolor="#2a3547",
    )
)


def chart_component(ohlcv_df=None, trades_df=None, symbol="NIFTYBEES-EQ", decisions_df=None):
    if ohlcv_df is None or ohlcv_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No OHLCV data — bot will populate after first trading cycle",
            showarrow=False,
            font=dict(color="#64748b", size=13),
        )
        fig.update_layout(**_LAYOUT, title=f"{symbol} — 5m")
        return dcc.Graph(figure=fig, style={"height": "420px"}, config={"displayModeBar": False})

    df = ohlcv_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Filter to NSE market hours only: 9:15 AM – 3:30 PM IST
    # yfinance includes overnight/pre-market candles with stale/wrong prices
    market_open  = pd.to_datetime("09:15").time()
    market_close = pd.to_datetime("15:31").time()
    df = df[df["timestamp"].dt.time.between(market_open, market_close)].reset_index(drop=True)

    # ── Remove price anomalies iteratively until stable
    # Catches both single-candle spikes AND entire bad days from yfinance
    for _ in range(3):
        if len(df) < 2:
            break
        median_price = df["close"].median()
        # Remove candles more than 5% away from the median price of the window
        bad_mask = (df["close"] - median_price).abs() / median_price > 0.05
        if bad_mask.sum() == 0:
            break
        df = df[~bad_mask].reset_index(drop=True)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No market-hours data available", showarrow=False,
                           font=dict(color="#64748b", size=13))
        fig.update_layout(**_LAYOUT, title=f"{symbol} — 5m OHLCV")
        return dcc.Graph(figure=fig, style={"height": "420px"}, config={"displayModeBar": False})

    # Indicators
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"]  = sma20 + 2 * std20
    df["bb_lower"]  = sma20 - 2 * std20
    df["bb_middle"] = sma20

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
        increasing_fillcolor="rgba(0,255,136,0.6)",
        decreasing_fillcolor="rgba(255,68,68,0.6)",
    ))

    # EMA 9
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["ema9"],
        mode="lines", name="EMA 9",
        line=dict(color="#4dabf7", width=1.5),
        hoverinfo="skip",
    ))

    # EMA 21
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["ema21"],
        mode="lines", name="EMA 21",
        line=dict(color="#a78bfa", width=1.5),
        hoverinfo="skip",
    ))

    # Bollinger Bands — fill between upper/lower
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["bb_upper"],
        mode="lines", name="BB Upper",
        line=dict(color="#4dabf7", width=0.5, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["bb_lower"],
        mode="lines", name="BB Lower",
        fill="tonexty",
        fillcolor="rgba(77,171,247,0.06)",
        line=dict(color="#4dabf7", width=0.5, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))

    # Trade markers — only show trades for this symbol
    if trades_df is not None and not trades_df.empty:
        trades_df = trades_df[trades_df["symbol"] == symbol]
    if trades_df is not None and not trades_df.empty:
        for _, trade in trades_df.iterrows():
            try:
                entry_t = pd.to_datetime(trade.get("entry_time"))
                exit_t  = pd.to_datetime(trade.get("exit_time"))
                ep = float(trade.get("entry_price", 0))
                xp = float(trade.get("exit_price", 0))
                pnl = float(trade.get("pnl", 0))
                strat = trade.get("strategy", "")

                fig.add_trace(go.Scatter(
                    x=[entry_t], y=[ep],
                    mode="markers",
                    marker=dict(size=10, color="#00ff88", symbol="triangle-up"),
                    hovertext=f"▲ ENTRY {strat}<br>₹{ep:.2f}",
                    hoverinfo="text", showlegend=False,
                ))

                if pd.notna(exit_t) and xp > 0:
                    fig.add_trace(go.Scatter(
                        x=[exit_t], y=[xp],
                        mode="markers",
                        marker=dict(size=10, color="#00ff88" if pnl >= 0 else "#ff4444", symbol="triangle-down"),
                        hovertext=f"▼ EXIT {strat}<br>₹{xp:.2f}  P&L: ₹{pnl:+,.0f}",
                        hoverinfo="text", showlegend=False,
                    ))
            except Exception:
                pass

    # AI decision markers — vetoes (red ✕) and approvals (green ★) for this symbol
    sym_bare = symbol.replace("-EQ", "").replace(".BO", "")
    if decisions_df is not None and not decisions_df.empty:
        # Match on full symbol OR bare name (e.g. "LT-EQ" or "LT")
        dec = decisions_df[
            decisions_df["symbol"].str.replace("-EQ", "").str.replace(".BO", "") == sym_bare
        ].copy()
        if not dec.empty:
            dec["decided_at"] = pd.to_datetime(dec["decided_at"], errors="coerce")
            dec = dec.dropna(subset=["decided_at"])
            approved = dec[dec["approved"] == 1]
            vetoed   = dec[dec["approved"] == 0]
            if not approved.empty:
                fig.add_trace(go.Scatter(
                    x=approved["decided_at"],
                    y=approved["signal_price"],
                    mode="markers",
                    name="AI Approved",
                    marker=dict(size=12, color="#00ff88", symbol="star",
                                line=dict(color="#ffffff", width=1)),
                    hovertext=approved.apply(
                        lambda r: f"✅ APPROVED {r.get('action','')}<br>"
                                  f"Strategy: {r.get('strategy','')}<br>"
                                  f"₹{r.get('signal_price',0):,.2f}  conf={r.get('signal_conf',0):.0%}",
                        axis=1),
                    hoverinfo="text",
                ))
            if not vetoed.empty:
                fig.add_trace(go.Scatter(
                    x=vetoed["decided_at"],
                    y=vetoed["signal_price"],
                    mode="markers",
                    name="AI Veto",
                    marker=dict(size=12, color="#ff4444", symbol="x",
                                line=dict(color="#ffffff", width=1)),
                    hovertext=vetoed.apply(
                        lambda r: f"🚫 VETO {r.get('action','')}<br>"
                                  f"Strategy: {r.get('strategy','')}<br>"
                                  f"₹{r.get('signal_price',0):,.2f}<br>"
                                  f"{str(r.get('veto_reason',''))[:80]}",
                        axis=1),
                    hoverinfo="text",
                ))

    # Default to today's trading session (09:15 today → latest candle)
    # Avoids the overnight flat-EMA artifact from a naive "last 24 hours" window
    last_ts = df["timestamp"].iloc[-1]
    today_open = last_ts.normalize() + pd.Timedelta(hours=9, minutes=15)
    # If last candle is before today's open (weekend/holiday), fall back to last trading day
    if last_ts < today_open:
        today_open = last_ts.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=9, minutes=15)
    first_1d = today_open

    layout = dict(**_LAYOUT, title=f"{symbol} — 5m OHLCV")
    layout["xaxis"] = dict(
        **_LAYOUT["xaxis"],
        **_RANGE_BUTTONS,
        range=[str(first_1d), str(last_ts + pd.Timedelta(minutes=30))],
    )
    fig.update_layout(**layout)

    return dcc.Graph(figure=fig, style={"height": "420px"}, config={"displayModeBar": False})
