"""Convenience launcher for the Modular RAG Dashboard.

Usage::

    python scripts/start_dashboard.py
    python scripts/start_dashboard.py --port 8502
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Modular RAG Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve the dashboard on")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
    parser.add_argument(
        "--demo", action="store_true", help="Use isolated model-free local_hash demo"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    app_path = root / "src" / "observability" / "dashboard" / "app.py"
    if not app_path.exists():
        print(f"Error: Dashboard app not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    print(f"Starting Dashboard: {' '.join(cmd)}")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    environment["STREAMLIT_SERVER_HEADLESS"] = "true"
    environment["STREAMLIT_THEME_PRIMARY_COLOR"] = "#166b58"
    if args.demo:
        environment["MODULAR_RAG_SETTINGS_PATH"] = str(root / "config/settings.demo.yaml")
    sys.exit(subprocess.run(cmd, cwd=root, env=environment).returncode)


if __name__ == "__main__":
    main()
