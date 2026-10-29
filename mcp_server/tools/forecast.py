"""HTTP client used by the MCP service to call the forecast API."""

import math
from typing import Any
import datetime

import httpx
from .responseSchema import *
from .charts import chart_block
from mcp.types import TextContent, EmbeddedResource

from .config import Settings


class ForecastApiError(RuntimeError):
    """A request to the forecast API could not be completed successfully."""


class ForecastApiClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.forecast_api_base_url
        self._timeout = settings.forecast_api_timeout_seconds

    async def health_check(self) -> list[TextContent]:
        return await self._request("GET", "/health")

    async def forecast(
        self, ticker: str, as_of_date: str
    ) -> ForecastResult | list[TextContent]:
        if len(as_of_date) == 0:
            as_of_date = datetime.date.today().strftime('%Y-%m-%d')
        res = await self._request(
            "POST",
            "/predict",
            json={
                "ticker": ticker,
                "as_of_date": as_of_date,
            },
        )
        return self._format_forecast(res)

    async def forecast_reasoning(
            self, ticker: str, as_of_date:str
    ) -> list[TextContent | EmbeddedResource]:
        if len(as_of_date) == 0:
            as_of_date = datetime.date.today().strftime('%Y-%m-%d')
        return self._format_forecast_reasoning(await self._request(
            "POST",
            "/predict",
            json={
                "ticker": ticker,
                "as_of_date": as_of_date,
            }
        ))

    async def plot_forecast(
                self, ticker: str, as_of_date:str
        ) -> list[TextContent | EmbeddedResource]:
            if len(as_of_date) == 0:
                as_of_date = datetime.date.today().strftime('%Y-%m-%d')
            return self._format_forecast_plot(await self._request(
                "POST",
                "/predict",
                json={
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                }
            ))

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            ) as client:
                response = await client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ForecastApiError(
                f"Forecast API timed out after {self._timeout:g} seconds."
            ) from exc
        except httpx.HTTPError as exc:
            raise ForecastApiError(f"Could not reach forecast API: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ForecastApiError("Forecast API returned an invalid JSON response.") from exc

        if response.is_error:
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            raise ForecastApiError(
                f"Forecast API returned HTTP {response.status_code}: {detail}"
            )

        if not isinstance(payload, dict):
            raise ForecastApiError("Forecast API returned a JSON value instead of an object.")
        return payload

    def _to_pct(self, log_return: float) -> float:
        return round((math.exp(log_return) - 1) * 100, 2)

    def _format_forecast(self, response: dict[str, Any]) -> ForecastResult | list[TextContent]:


        forecast = response["forecast"]

        REPORT_HORIZONS = (5, 10, 15, 20)
        forecasts = []
        median_returns = []

        for day in REPORT_HORIZONS:
            index = day - 1
            q10 = self._to_pct(forecast["q10"][index])
            q50 = self._to_pct(forecast["q50"][index])
            q90 = self._to_pct(forecast["q90"][index])

            forecasts.append(f"day {day}: q10={q10}%, q50={q50}%, q90={q90}%")
            median_returns.append(q50)
            if day == 20:
                expected_return_pct = q50
                uncertainty_width = round(q90 - q10, 2)

        ridge = response["ridge"]
        ridge_forecasts = [
            f"day {day}: {self._to_pct(value)}%"
            for day, value in zip(ridge["horizon_days"], ridge["prediction"])
        ]

        if expected_return_pct > 0 and median_returns == sorted(median_returns):
            trend = "gradually_positive"
        elif expected_return_pct < 0 and median_returns == sorted(median_returns, reverse=True):
            trend = "gradually_negative"
        else:
            trend = "mixed"

        return ForecastResult(
            Ticker=response["ticker"],
            Horizon_Days=len(forecast["horizon_days"]),
            forecasts=forecasts,
            ridge_forecasts=ridge_forecasts,
            expected_return_pct=expected_return_pct,
            uncertainty_width_pct=uncertainty_width,
            trend=trend
        )

    def _format_forecast_reasoning(self, response: dict[str, Any]) -> list[TextContent | EmbeddedResource]:


        feature_importance = response["feature_importance"]
        attention = response["attention"]

        top_n_features = 3 # number of features to include in the reasoning summary
        top_n_attention_days = 5 # number of days to include in the reasoning summary
        k = 10 # number of most recent trading sessions to consider for recency bias score (should be tuned post training)

        feature_importance_dict = {}
        feature_importance["weights"] = [round(weight, 4) for weight in feature_importance["weights"]]
        for feature, weight in zip(feature_importance["features"], feature_importance["weights"]): feature_importance_dict[feature] = weight
        top_features = list(sorted(feature_importance_dict.items(), key=lambda f: f[1], reverse=True))[:top_n_features]

        attention_dict = {}
        attention["weights"] = [round(weight, 4) for weight in attention["weights"]]
        for day, weight in zip(attention["timesteps"], attention["weights"]): attention_dict[day] = weight
        attention_sorted = list(sorted(attention_dict.items(), key=lambda f: f[1], reverse=True))[:top_n_attention_days]

        recency_bias_score = round(sum(attention["weights"][-k:]), 4)

        reasoning = ForecastReasoning(
            Ticker=response["ticker"],
            Highest_Feature_importance=[f"{feature}: {weight}" for feature, weight in top_features],
            Highest_Attention_Days=[f"{day}: {weight}" for day, weight in attention_sorted],
            Recency_Bias_Score=recency_bias_score
        )

        top_n_chart_features = 8
        chart = chart_block(
            "reasoning",
            response["ticker"],
            features=[
                {"name": feature, "weight": weight}
                for feature, weight in sorted(feature_importance_dict.items(), key=lambda f: f[1], reverse=True)[:top_n_chart_features]
            ],
            attention=[{"date": day, "weight": weight} for day, weight in attention_dict.items()],
            recency_days=k,
            recency_score=recency_bias_score,
        )
        return [
            TextContent(type="text", text=reasoning.model_dump_json()),
            TextContent(type="text", text="Feature importance and attention charts are displayed to the user."),
            chart,
        ]

    def _format_forecast_plot(self, response: dict[str, Any]) -> list[TextContent | EmbeddedResource]:
        forecast = response["forecast"]
        result = self._format_forecast(response)

        summary = f"Forecast Summary for {result.Ticker} (returns are excess over SPY):\n" \
                f"Expected Return (20-day): {result.expected_return_pct}%\n" \
                f"Uncertainty Width (20-day): {result.uncertainty_width_pct}%\n" \
                f"Trend: {result.trend}\n" \
                "A forecast chart is displayed to the user."

        chart = chart_block(
            "forecast",
            result.Ticker,
            as_of=response["encoder_end_date"],
            start=response["forecast_start_date"],
            days=forecast["horizon_days"],
            q10=[self._to_pct(v) for v in forecast["q10"]],
            q50=[self._to_pct(v) for v in forecast["q50"]],
            q90=[self._to_pct(v) for v in forecast["q90"]],
        )
        return [TextContent(type="text", text=summary), chart]
