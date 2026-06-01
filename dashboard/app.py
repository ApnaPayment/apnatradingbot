"""
AlgoTrader Dashboard — Plotly Dash Web Interface
6-page trading dashboard for real-time algo trading monitoring.

Usage:
    python dashboard/app.py              # http://localhost:8050
    python dashboard/app.py --port 8080  # custom port
    python dashboard/app.py --debug      # debug mode
"""

import dash
import dash_bootstrap_components as dbc
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.layout import create_layout
from dashboard import callbacks  # Import to register callbacks


def create_app(debug=False, port=8050):
    """Create and return the Dash app."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.CYBORG,
            "/assets/style.css",
        ],
        suppress_callback_exceptions=True,
    )

    app.title = "AlgoTrader Dashboard"
    app.layout = create_layout()

    return app, port, debug


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AlgoTrader Dash Dashboard"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the server on (default: 8050)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (hot reload, verbose logging)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    app, port, debug = create_app(debug=args.debug, port=args.port)

    print(f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                   AlgoTrader Dashboard                         ║
    ║              Plotly Dash Real-Time Trading UI                  ║
    ║                                                                ║
    ║  🌐 http://{args.host}:{port}                    ║
    ║  📊 Pages: Command Center, Charts, Trades, Analytics, Risk   ║
    ║  🔄 Auto-refresh: 3s (positions), 60s (analytics)            ║
    ║  💾 Data: SQLite market_data.db                               ║
    ║                                                                ║
    ║  Press Ctrl+C to stop                                         ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    app.run(
        debug=debug,
        host=args.host,
        port=port,
    )


if __name__ == "__main__":
    main()
