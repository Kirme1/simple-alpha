"""WebSocket server exposing the AgentHarness to the frontend.

Run with:
    uv run uvicorn llm_harness.ws_server:app --port 8002
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import ALLOWED_ORIGINS, LLM_URL, MCP_URL
from .harness import AgentHarness
from .models import LOCAL_ID, available_models, build_provider

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared MCP session for all connections (the MCP server is stateless)
    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            app.state.mcp_session = mcp_session
            yield


app = FastAPI(title="Agent Harness WS", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
async def models() -> dict:
    return {"models": [m.public() for m in available_models()], "default": LOCAL_ID}


@app.websocket("/ws")
async def chat_socket(websocket: WebSocket) -> None:
    # CORS does not cover WebSockets, so without this check any web page could drive the chat (and spend the API keys).
    # Browsers always send Origin; clients that omit it are not cross-site browser pages.
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        logger.warning("Rejected WebSocket from origin %s", origin)
        await websocket.close(code=1008)
        return
    await websocket.accept()
    options = {m.id: m for m in available_models()}
    harness: AgentHarness | None = None
    harness_model: str | None = None
    mcp_session: ClientSession = websocket.app.state.mcp_session

    await websocket.send_json({
        "type": "models",
        "models": [m.public() for m in options.values()],
        "default": LOCAL_ID,
    })

    try:
        while True:
            message = await websocket.receive_json()

            if message.get("type") == "reset":
                harness = None
                continue

            if message.get("type") != "query" or not str(message.get("text", "")).strip():
                await websocket.send_json({"type": "error", "message": "Expected {type: 'query', text: '...'}"})
                continue

            model_id = message.get("model") or LOCAL_ID
            if model_id not in options:
                await websocket.send_json({"type": "error", "message": f"Model '{model_id}' is not available."})
                continue

            # Conversation history is provider-specific, so switching model starts a fresh one.
            if harness is None or harness_model != model_id:
                harness = AgentHarness(build_provider(options[model_id], LLM_URL))
                harness_model = model_id

            await websocket.send_json({"type": "status", "state": "thinking"})
            try:
                answer = await harness.query(
                    mcp_session,
                    message["text"].strip(),
                    on_event=websocket.send_json,
                )
                await websocket.send_json({"type": "answer", "text": answer})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logger.exception("Query failed")
                await websocket.send_json({"type": "error", "message": "The request failed. See the harness logs for details."})
    except WebSocketDisconnect:
        logger.info("Client disconnected")
