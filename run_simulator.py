#!/usr/bin/env python3
"""
Launcher for the Ring AI • Ultrahuman Readiness Score Interactive Simulator.
Ensures the API is healthy and accepting connections before launching the browser.
"""
import argparse
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn


def check_api_health(health_url: str, timeout: float = 0.8) -> bool:
    """Check if the API health endpoint is responding with 200 OK."""
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "RingAI-Launcher"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_and_open_browser(simulator_url: str, health_url: str, max_retries: int = 40, interval: float = 0.25):
    """
    Polls the health endpoint in a background thread until the server is fully ready,
    then automatically opens the simulator in the default browser.
    Prevents blank/connection-refused pages on initial load.
    """
    for _ in range(max_retries):
        time.sleep(interval)
        if check_api_health(health_url):
            try:
                webbrowser.open(simulator_url)
            except Exception:
                pass
            return


def main():
    parser = argparse.ArgumentParser(description="Ring AI Readiness Score Simulator Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    parser.add_argument("--open-only", action="store_true", help="Only open the browser assuming the API is already running")
    args = parser.parse_args()

    simulator_url = f"http://{args.host}:{args.port}/simulator"
    docs_url = f"http://{args.host}:{args.port}/docs"
    health_url = f"http://{args.host}:{args.port}/health"

    # Mode 1: --open-only flag
    if args.open_only:
        print(f"Opening simulator in browser: {simulator_url}")
        webbrowser.open(simulator_url)
        return

    # Check if API is already running
    if check_api_health(health_url):
        print("=" * 70)
        print("  RING AI: API IS ALREADY RUNNING AND READY")
        print("=" * 70)
        print(f"  Interactive Simulator:  {simulator_url}")
        print(f"  FastAPI Swagger Docs:   {docs_url}")
        print(f"  Health Check:           {health_url}")
        print("=" * 70)
        if not args.no_browser:
            print("  Opening simulator in your default browser...")
            webbrowser.open(simulator_url)
        return

    print("=" * 70)
    print("  RING AI: ULTRAHUMAN READINESS SCORE SIMULATOR")
    print("=" * 70)
    print(f"  Interactive Simulator:  {simulator_url}")
    print(f"  FastAPI Swagger Docs:   {docs_url}")
    print(f"  Health Check:           {health_url}")
    print("=" * 70)
    print("  1. Initializing API server & loading model artifacts...")
    if not args.no_browser:
        print("  2. Browser will open automatically once the API is verified healthy.")
    print("  Press Ctrl+C to terminate the server.")
    print("=" * 70)

    # Launch background thread to wait for health check before opening browser
    if not args.no_browser:
        watcher_thread = threading.Thread(
            target=wait_and_open_browser,
            args=(simulator_url, health_url),
            daemon=True
        )
        watcher_thread.start()

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
