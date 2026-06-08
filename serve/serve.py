"""
Ollama serving startup script.
Detects ollama, starts the server, pulls the model, and confirms readiness.
"""

import subprocess
import sys
import time
import shutil
import os
import requests

MODEL = "qwen2.5:1.5b"
OLLAMA_HOST = "http://localhost:11434"
COMMON_PATHS = [
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
    r"C:\Program Files\Ollama\ollama.exe",
    shutil.which("ollama"),
]


def find_ollama() -> str:
    for p in COMMON_PATHS:
        if p and os.path.isfile(p):
            return p
    print("ERROR: ollama not found. Install from https://ollama.com/download")
    sys.exit(1)


def server_healthy() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def start_server(ollama_bin: str):
    if server_healthy():
        print("ollama server already running.")
        return

    print("Starting ollama serve …")
    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    for i in range(30):
        time.sleep(1)
        if server_healthy():
            print("ollama server is up.")
            return
    print("ERROR: ollama server did not start within 30 s.")
    sys.exit(1)


def pull_model(ollama_bin: str):
    tags = requests.get(f"{OLLAMA_HOST}/api/tags").json()
    names = [m["name"] for m in tags.get("models", [])]
    if any(MODEL in n for n in names):
        print(f"Model {MODEL} already pulled.")
        return

    print(f"Pulling {MODEL} (this may take a few minutes) …")
    subprocess.run([ollama_bin, "pull", MODEL], check=True)
    print(f"Model {MODEL} ready.")


def main():
    ollama_bin = find_ollama()
    print(f"Found ollama at {ollama_bin}")

    start_server(ollama_bin)
    pull_model(ollama_bin)

    print(f"\n=== ollama serving {MODEL} at {OLLAMA_HOST} ===")
    print(f"  Generate endpoint: {OLLAMA_HOST}/api/generate")
    print(f"  OpenAI compat:     {OLLAMA_HOST}/v1/completions")


if __name__ == "__main__":
    main()
