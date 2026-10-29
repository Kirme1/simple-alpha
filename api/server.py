import logging
from contextlib import asynccontextmanager
from .config import CHECKPOINT_PATH

from fastapi import FastAPI, HTTPException

from .model_service import ModelService
from ML_layer.framework.ridge_forecaster import RIDGE_HORIZONS
from .schemas import (
    AttentionPattern,
    FeatureImportance,
    ForecastTrajectory,
    InferenceRequest,
    InferenceResponse,
    RidgeForecast,
)

logger = logging.getLogger(__name__)

_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    logger.info("Loading model from %s ...", CHECKPOINT_PATH)
    _service = ModelService()
    _service.load()
    logger.info("Models loaded and calibrated. Ready to serve.")
    yield
    _service = None


app = FastAPI(title="PTF-TFT Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _service is not None}


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest) -> InferenceResponse:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = _service.predict(request.ticker, str(request.as_of_date))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    horizon = len(result["q50"])

    return InferenceResponse(
        ticker=request.ticker,
        encoder_end_date=result["encoder_end_date"],
        forecast_start_date=result["encoder_end_date"],
        forecast=ForecastTrajectory(
            horizon_days=list(range(1, horizon + 1)),
            q10=result["q10"],
            q50=result["q50"],
            q90=result["q90"],
        ),
        ridge=RidgeForecast(
            horizon_days=list(RIDGE_HORIZONS),
            prediction=result["ridge"],
        ),
        feature_importance=FeatureImportance(
            features=result["encoder_var_names"],
            weights=result["encoder_vars"],
        ),
        attention=AttentionPattern(
            timesteps=result["encoder_dates"],
            weights=result["attention"],
        ),
    )
