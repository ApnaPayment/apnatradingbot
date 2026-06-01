"""Layout — Sidebar, page router, multi-page structure."""

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback
import os

SIDEBAR_BG   = "#151a24"
SIDEBAR_LINK = "#94a3b8"
ACTIVE_BG    = "rgba(77,171,247,0.15)"
ACTIVE_COLOR = "#4dabf7"

NAV_LINKS = [
    ("🎮 Command Center", "/",          "link-command-center"),
    ("📈 Charts",         "/charts",     "link-charts"),
    ("📋 Trades",         "/trades",     "link-trades"),
    ("📊 Analytics",      "/analytics",  "link-analytics"),
    ("⚠️  Risk",           "/risk",       "link-risk"),
    ("⚙️  Settings",       "/settings",   "link-settings"),
]


def create_sidebar():
    """Left sidebar with page links."""
    nav_items = []
    for label, href, link_id in NAV_LINKS:
        nav_items.append(
            dcc.Link(
                html.Div(
                    label,
                    id=link_id,
                    style={
                        "padding": "10px 12px",
                        "border-radius": "4px",
                        "color": SIDEBAR_LINK,
                        "font-size": "12px",
                        "cursor": "pointer",
                        "transition": "all 0.2s ease",
                    },
                ),
                href=href,
                style={"text-decoration": "none"},
            )
        )

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H5(
                        "📊 AlgoTrader",
                        style={
                            "color": "#4dabf7",
                            "margin": "0 0 24px 0",
                            "font-size": "14px",
                            "font-weight": "700",
                            "text-transform": "uppercase",
                            "letter-spacing": "0.5px",
                        },
                    ),
                    html.Div(
                        children=nav_items,
                        style={"display": "flex", "flex-direction": "column", "gap": "4px"},
                    ),
                ],
                style={"padding": "16px"},
            ),
        ],
        style={
            "background-color": SIDEBAR_BG,
            "border-right": "2px solid #0d0f14",
            "width": "200px",
            "height": "100vh",
            "position": "fixed",
            "left": "0",
            "top": "0",
            "overflow-y": "auto",
        },
    )


def create_layout():
    """Main app layout with sidebar and page router."""
    return html.Div(
        children=[
            dcc.Location(id="url", refresh=False),
            html.Div(
                children=[
                    create_sidebar(),
                    html.Div(
                        id="page-content",
                        style={
                            "margin-left": "200px",
                            "width": "calc(100% - 200px)",
                            "min-height": "100vh",
                            "background-color": "#0d0f14",
                        },
                    ),
                ],
                style={"display": "flex", "width": "100%", "height": "100%"},
            ),
            dcc.Interval(id="interval-component", interval=3000, n_intervals=0),
            # Hidden download trigger for trades CSV
            dcc.Download(id="trades-download"),
        ],
        style={
            "background-color": "#0d0f14",
            "color": "#e2e8f0",
            "font-family": "'Inter', 'Roboto', sans-serif",
            "margin": "0",
            "padding": "0",
        },
    )


# ── Active sidebar link highlight ─────────────────────────────────────────────
@callback(
    [Output(link_id, "style") for _, _, link_id in NAV_LINKS],
    Input("url", "pathname"),
)
def highlight_active_link(pathname):
    styles = []
    for _, href, _ in NAV_LINKS:
        is_active = (pathname == href) or (href != "/" and (pathname or "").startswith(href))
        styles.append({
            "padding": "10px 12px",
            "border-radius": "4px",
            "color": ACTIVE_COLOR if is_active else SIDEBAR_LINK,
            "background-color": ACTIVE_BG if is_active else "transparent",
            "font-size": "12px",
            "font-weight": "700" if is_active else "400",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
        })
    return styles
