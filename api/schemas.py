from pydantic import BaseModel
from datetime import date


class InferenceRequest(BaseModel):
    ticker: str
    # Latest date whose market data may be used by the encoder. The service
    # selects the preceding 60 available trading rows automatically.
    as_of_date: date


class ForecastTrajectory(BaseModel):
    horizon_days: list[int]  # [1, 2, ..., 20]
    q10: list[float]         # 10th percentile cumulative log return over SPY per horizon
    q50: list[float]         # point prediction
    q90: list[float]         # 90th percentile


class RidgeForecast(BaseModel):
    horizon_days: list[int]  # the few horizons the ridge model predicts, e.g. [1, 5, 20]
    prediction: list[float]  # cumulative log return over SPY per horizon (point forecast)


class FeatureImportance(BaseModel):
    features: list[str]   # feature names (encoder variables of the model)
    weights: list[float]  # VSN encoder variable selection weights, sum to ~1


class AttentionPattern(BaseModel):
    timesteps: list[str]  # date strings for each encoder position
    weights: list[float]  # attention weight per timestep, sum to ~1


class InferenceResponse(BaseModel):
    ticker: str
    encoder_end_date: str
    forecast_start_date: str
    forecast: ForecastTrajectory
    ridge: RidgeForecast
    feature_importance: FeatureImportance
    attention: AttentionPattern
