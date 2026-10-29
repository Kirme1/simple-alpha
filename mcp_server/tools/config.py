"""Runtime configuration for the MCP service."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Settings supplied through environment variables."""

    forecast_api_base_url: str = "http://127.0.0.1:8000"
    forecast_api_timeout_seconds: float = 60.0
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            forecast_api_base_url=os.getenv(
                "FORECAST_API_BASE_URL", "http://127.0.0.1:8000"
            ).rstrip("/"),
            forecast_api_timeout_seconds=float(
                os.getenv("FORECAST_API_TIMEOUT_SECONDS", "60")
            ),
            mcp_host=os.getenv("MCP_HOST", "127.0.0.1"),
            mcp_port=int(os.getenv("MCP_PORT", "8001")),
        )
