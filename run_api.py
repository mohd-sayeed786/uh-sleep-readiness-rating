#!/usr/bin/env python3
"""
Start the Ring AI FastAPI production service.
"""
import sys
import socket
import urllib.request
from pathlib import Path

# Ensure project root is in python path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn


def is_server_already_running(port: int = 8000) -> bool:
    """Check if port 8000 is listening and responds to /health."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"User-Agent": "RingAI-Launcher"})
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            return resp.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000

    print("=" * 65)
    print("  RING AI: READINESS SCORE API SERVICE")
    print("=" * 65)
    print(f"  Interactive Swagger Docs:  http://localhost:{port}/docs")
    print(f"  Ultrahuman UI Simulator:   http://localhost:{port}/simulator")
    print(f"  System Health Check:       http://localhost:{port}/health")
    print("=" * 65)

    if is_server_already_running(port):
        print("  Notice: Service is already running and active on port 8000!")
        print("  You can open the URLs above in your browser right away.\n")
        sys.exit(0)

    from src.config import MODEL_PATH
    if not MODEL_PATH.exists():
        print("  Notice: Model artifact not found. Training model at runtime from raw data...")
        from src.train import train_model
        train_model(save_artifacts=True)
        print("  Champion model trained successfully!\n")

    print("  Starting server... Press Ctrl+C to terminate.\n")
    uvicorn.run("src.api:app", host=host, port=port, reload=False)
