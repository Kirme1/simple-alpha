from pydantic import BaseModel
import datetime

class ForecastResult(BaseModel):
    Ticker: str
    Horizon_Days: int
    forecasts: list[str]
    ridge_forecasts: list[str]
    expected_return_pct: float
    uncertainty_width_pct: float
    trend: str

class ForecastReasoning(BaseModel):
    Ticker: str
    Highest_Feature_importance: list[str]
    Highest_Attention_Days: list[str]
    Recency_Bias_Score: float

class PriceHistory(BaseModel):
    Ticker: str
    Trading_Days: int
    Start_Date: datetime.date
    End_Date: datetime.date
    Start_Close: float
    End_Close: float
    period_return_pct: float
    max_drawdown_pct: float
    period_return_pct: float
    avg_daily_volume: float

class MarketNews(BaseModel):
    Ticker: str
    News_Items: list[dict[str, str]]

class ArticleContent(BaseModel):
    Url: str
    Title: str
    Content: str
    Truncated: bool