"""
Entry point: starts FastAPI (port 8000) and Streamlit (port 8501) together.

Usage:
    python main.py

Or run each server separately:
    uvicorn chatbot.app:app --reload --port 8000
    streamlit run chatbot/ui.py --server.port 8501
"""
import subprocess
import sys
import threading

from config import FASTAPI_PORT, STREAMLIT_PORT


def _run_fastapi() -> None:
    subprocess.run([
        sys.executable, "-m", "uvicorn", "chatbot.app:app",
        "--host", "0.0.0.0",
        "--port", str(FASTAPI_PORT),
        "--reload",
    ])


def _run_streamlit() -> None:
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "chatbot/ui.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
    ])


if __name__ == "__main__":
    print(f"Starting FastAPI  → http://localhost:{FASTAPI_PORT}")
    print(f"Starting Streamlit → http://localhost:{STREAMLIT_PORT}\n")

    api_thread = threading.Thread(target=_run_fastapi, daemon=True)
    api_thread.start()

    _run_streamlit()
