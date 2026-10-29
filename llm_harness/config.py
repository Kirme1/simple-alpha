"""Service URLs for the harness, read from the environment (loaded from ``.env`` if present)."""

import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to .env (see DEVELOPMENT.md).")
    return value


MCP_URL = _required("MCP_URL")
LLM_URL = _required("LLM_URL")

# Browser origins allowed to open the chat socket (comma separated). Add yours when serving the UI from another address.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
