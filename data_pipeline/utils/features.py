"""Close-of-day feature engineering. Nothing here looks ahead of the row date."""
import numpy as np
import pandas as pd

from .calendars import CPI_RELEASE_DATES, FOMC_DATES
from .fred import FRED_FEATURES, FRED_LEVEL_SUFFIX
from ..yahoo import SECTOR_NAME_MAPPINGS, normalise_index


# Earnings-event features need Yahoo earnings dates (see fetch_earnings).
EARNINGS_FEATURES = ["days_since_earnings", "earnings_surprise"]

# Trading days after which days_since_earnings stops increasing (about a quarter).
EARNINGS_DAYS_CAP = 63

# Features from daily open/high/low/close: overnight vs intraday returns, extreme
# days, quantile skewness of daily returns, and close location within the day range.
PRICE_ACTION_FEATURES = [
    "overnight_ret_20d", "overnight_ret_60d", "intraday_ret_20d", "intraday_ret_60d",
    "max1_20d", "max5_20d", "min1_20d",
    "skew_q_60d", "skew_q_126d", "skew_q_252d",
    "clv_5d", "clv_20d",
]

# Rolling betas of a stock's daily return to daily changes in a macro series, so a
# market-wide FRED series becomes a per-stock feature that survives ranking.
# feature -> (FRED column, how the daily change is taken).
MACRO_BETA_WINDOW = 252
MACRO_BETA_MIN_PERIODS = 180
MACRO_BETAS = {
    "beta_term_252d": ("fred_term_spread_3m", "diff"),
    "beta_oil_252d": ("fred_oil", "log"),
    "beta_default_252d": ("fred_default_spread", "diff"),
}
MACRO_BETA_FEATURES = list(MACRO_BETAS)

# Market-wide z-scores times a stock characteristic, so they differ across tickers.
INTERACTION_FEATURES = ["vix_x_reversal", "vix_x_momentum"]

# Per-stock z-scores against the stock's own past year. Cross-sectional ranking
# leaves them alone: they keep the magnitude and own-history information it discards.
TS_ZSCORE_WINDOW = 252
TS_ZSCORE_MIN_PERIODS = 126
TS_NORMALISED_SOURCES = ["volatility_20d", "volume_rel_20d", "log_return_20d", "beta_mkt"]
TS_NORMALISED_FEATURES = [f"{c}_z252" for c in TS_NORMALISED_SOURCES]

# Input features are separate from the price-level target.
OBSERVED_FEATURES = [
    "log_return_1d", "log_return_5d", "log_return_20d", "log_return_60d",
    "volume_rel_20d", "high_ratio", "low_ratio", "rsi", "volatility_20d", "price_to_ma50",
    "spy_ret_20d", "sector_vs_market", "stock_vs_sector", "beta_mkt", "beta_sector",
    "momentum_12_1", "dist_52w_high", "idio_vol_60d",
    *PRICE_ACTION_FEATURES,
    *EARNINGS_FEATURES,
    *MACRO_BETA_FEATURES,
    *INTERACTION_FEATURES,
    *TS_NORMALISED_FEATURES,
]
KNOWN_REAL_FEATURES = ["days_to_month_end", "is_fomc_day", "is_cpi_release_day"]
KNOWN_CATEGORICAL_FEATURES = ["day_of_week", "month"]


# Inputs that take the same value for every ticker on a date. They cannot rank
# tickers by themselves, so cross-sectional ranking leaves them unchanged.
MARKET_WIDE_FEATURES = ["spy_ret_20d", *FRED_FEATURES, *KNOWN_REAL_FEATURES, *KNOWN_CATEGORICAL_FEATURES]
UNRANKED_FEATURES = [*MARKET_WIDE_FEATURES, *TS_NORMALISED_FEATURES]


def cross_sectional_rank(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace each column by its percentile rank among tickers on the same date, centred on 0.

    Each value is compared only with other tickers' values on its own date, so
    ranking introduces no look-ahead. Market-wide and time-series-normalised
    columns are left unchanged.
    """
    columns = [c for c in columns if c not in UNRANKED_FEATURES]
    ranked = df.groupby("date")[columns].rank(pct=True) - 0.5
    return df.assign(**{c: ranked[c] for c in columns})


def add_earnings_features(index: pd.DatetimeIndex, earnings: pd.DataFrame) -> pd.DataFrame:
    """Trading days since the last announcement and its EPS surprise, point-in-time.

    Features are close-of-day values, so an announcement counts on its own date
    only when released before the 16:00 close; later ones count from the next
    trading day.
    """
    index = pd.DatetimeIndex(index)
    announced = earnings["announced_at"]
    available_from = announced.dt.normalize() + pd.to_timedelta((announced.dt.hour >= 16).astype(int), unit="D")


    # Position of the first trading day on or after each announcement becomes known.
    positions = index.searchsorted(available_from.to_numpy())
    events = pd.DataFrame({"position": positions, "surprise": earnings["surprise"].fillna(0.0).to_numpy()})
    events = events[events["position"] < len(index)].drop_duplicates("position", keep="last")

    out = pd.DataFrame(index=index, columns=EARNINGS_FEATURES, dtype=float)

    last_position = pd.Series(np.nan, index=range(len(index)))
    last_position[events["position"].to_numpy()] = events["position"].to_numpy()
    last_surprise = pd.Series(np.nan, index=range(len(index)))
    last_surprise[events["position"].to_numpy()] = events["surprise"].to_numpy()
    days_since = np.arange(len(index)) - last_position.ffill().to_numpy()

    out["days_since_earnings"] = np.minimum(days_since, EARNINGS_DAYS_CAP)
    out["earnings_surprise"] = last_surprise.ffill().clip(-1.0, 1.0).to_numpy()
    return out


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:

    dates = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()

    out = pd.DataFrame(index=dates)
    out["day_of_week"] = dates.dayofweek.astype(str)
    out["month"] = dates.month.astype(str)
    out["days_to_month_end"] = (dates.days_in_month - dates.day).astype("int16")

    date_strings = dates.strftime("%Y-%m-%d")
    out["is_fomc_day"] = date_strings.isin(FOMC_DATES).astype("int8")
    out["is_cpi_release_day"] = date_strings.isin(CPI_RELEASE_DATES).astype("int8")

    return out


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Open/close/range features from split- and dividend-adjusted OHLC (``auto_adjust=True``).

    Overnight is log(Open_t / Close_{t-1}) and intraday is log(Close_t / Open_t), so
    they sum to the daily log return. Every window ends at day t.
    """

    out = pd.DataFrame(index=df.index)
    daily = np.log(df["Close"]).diff()
    overnight = np.log(df["Open"] / df["Close"].shift(1))
    intraday = np.log(df["Close"] / df["Open"])


    for days in (20, 60):
        out[f"overnight_ret_{days}d"] = overnight.rolling(days).sum()
        out[f"intraday_ret_{days}d"] = intraday.rolling(days).sum()
    
    out["max1_20d"] = daily.rolling(20).max()
    # Mean of the five highest daily returns in the window.
    out["max5_20d"] = daily.rolling(20).apply(lambda w: np.sort(w)[-5:].mean(), raw=True)
    out["min1_20d"] = daily.rolling(20).min()


    for days in (60, 126, 252):
        window = daily.rolling(days)
        q10, q50, q90 = window.quantile(0.1), window.quantile(0.5), window.quantile(0.9)
        out[f"skew_q_{days}d"] = (q90 + q10 - 2 * q50) / (q90 - q10).replace(0, np.nan)


    day_range = df["High"] - df["Low"]
    clv = ((df["Close"] - df["Low"]) / day_range.replace(0, np.nan)).fillna(0.5).clip(0, 1)


    for days in (5, 20):
        out[f"clv_{days}d"] = clv.rolling(days).mean()

    return out[PRICE_ACTION_FEATURES]


def engineer_features(
    df: pd.DataFrame,
    sector: str,
    *,
    include_calendar: bool = True,
    earnings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build close-of-day features. Missing context is never backward-filled.

    Earnings features are only added when ``earnings`` (from fetch_earnings) is given.
    """

    df = normalise_index(df)

    sector_etf = SECTOR_NAME_MAPPINGS.get(sector)

    if sector_etf is None:
        raise ValueError(f"No sector ETF mapping for {sector!r}")
    missing = {"Open", "High", "Low", "Close", "Volume", "SPY", sector_etf}.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required feature inputs: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)

    log_close = np.log(df["Close"])
    for days in (1, 5, 20, 60):
        out[f"log_return_{days}d"] = log_close.diff(days)
    out["volume_rel_20d"] = df["Volume"] / df["Volume"].rolling(20).mean() - 1.0
    out["high_ratio"] = df["High"] / df["Close"] - 1.0
    out["low_ratio"] = df["Low"] / df["Close"] - 1.0


    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = gain / loss.replace(0, np.nan)


    out["rsi"] = 100 - 100 / (1 + relative_strength)
    out.loc[(loss == 0) & (gain > 0), "rsi"] = 100.0
    out["volatility_20d"] = log_close.diff().rolling(20).std() * np.sqrt(252)
    out["price_to_ma50"] = df["Close"] / df["Close"].rolling(50).mean() - 1.0


    sector_ret = np.log(df[sector_etf]).diff(20)
    spy_ret = np.log(df["SPY"]).diff(20)
    stock_ret = log_close.diff(20)

    out["spy_ret_20d"] = spy_ret
    out["sector_vs_market"] = sector_ret - spy_ret
    out["stock_vs_sector"] = stock_ret - sector_ret
    out["beta_mkt"], out["beta_sector"] = rolling_two_factor_beta(stock_ret, spy_ret, sector_ret)

    # 12-month return skipping the latest month, whose short-term reversal
    # would otherwise offset momentum.
    out["momentum_12_1"] = log_close.shift(21) - log_close.shift(252)
    out["dist_52w_high"] = df["Close"] / df["High"].rolling(252).max() - 1.0
    out["idio_vol_60d"] = rolling_idiosyncratic_volatility(
        log_close.diff(), np.log(df["SPY"]).diff(), np.log(df[sector_etf]).diff(),
    )

    out = out.join(add_price_action_features(df))
    for source in TS_NORMALISED_SOURCES:
        out[f"{source}_z252"] = rolling_zscore(out[source])

    out = out.join(add_macro_features(out, df))

    if earnings is not None:
        out = out.join(add_earnings_features(out.index, earnings))

    for column in FRED_FEATURES:
        if column in df:
            out[column] = df[column]
    out["target_log_close"] = log_close

    # Targets are excess returns over SPY; keep its log close alongside.
    out["target_spy_log_close"] = np.log(df["SPY"])

    if include_calendar:
        out = out.join(add_calendar_features(out.index))

    return out.replace([np.inf, -np.inf], np.nan).dropna()


def rolling_zscore(x: pd.Series, window: int = TS_ZSCORE_WINDOW, min_periods: int = TS_ZSCORE_MIN_PERIODS) -> pd.Series:
    """Z-score of ``x`` against its own trailing window, which ends at (and includes) day t."""
    rolling = x.rolling(window, min_periods=min_periods)
    return (x - rolling.mean()) / rolling.std().replace(0, np.nan)


def macro_change(level: pd.Series, how: str) -> pd.Series:
    """Daily change of a macro level: a difference, or a log change for price-like series."""
    if how == "log":
        # WTI printed a negative price in April 2020, where a log change is undefined.
        return np.log(level.where(level > 0)).diff()
    return level.diff()


def rolling_macro_beta(
    stock_ret: pd.Series,
    macro_delta: pd.Series,
    window: int = MACRO_BETA_WINDOW,
    min_periods: int = MACRO_BETA_MIN_PERIODS,
) -> pd.Series:
    """Rolling slope of daily stock returns on daily macro changes, over the window ending at day t.

    FRED values on row t were released before t, so ``macro_delta`` on row t is a
    move that ended by t-1. The stock return is lagged one row to cover the same
    day; no return or macro value after day t-1's close enters the beta at t.
    """

    y = stock_ret.shift(1)
    # Common sample, so the variance and covariance see the same days.
    valid = y.notna() & macro_delta.notna()
    y, x = y.where(valid), macro_delta.where(valid)
    covariance = y.rolling(window, min_periods=min_periods).cov(x)
    return covariance / x.rolling(window, min_periods=min_periods).var().replace(0, np.nan)


def add_macro_features(out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Per-stock macro betas and macro x stock interactions; empty when FRED is absent.

    ``out`` holds the stock features built so far and ``df`` the joined inputs
    (FRED z-scores and ``*_level`` columns).
    """

    feats = pd.DataFrame(index=out.index)
    daily_ret = out["log_return_1d"]

    for name, (column, how) in MACRO_BETAS.items():
        level = df.get(column + FRED_LEVEL_SUFFIX)
        if level is not None:
            feats[name] = rolling_macro_beta(daily_ret, macro_change(level, how))

    if "fred_vix" in df:
        feats["vix_x_reversal"] = df["fred_vix"] * -out["log_return_5d"]
        feats["vix_x_momentum"] = df["fred_vix"] * out["momentum_12_1"]

    return feats


def _rolling_two_factor_fit(stock_ret, spy_ret, sector_ret, window: int, intercept: bool):
    """Per-row OLS of stock on market and sector-excess returns over the prior ``window`` rows.

    Returns (betas [N, 2], residual standard deviation [N]).
    """
    
    sector_excess = np.asarray(sector_ret - spy_ret, dtype=float)
    X = np.column_stack([np.asarray(spy_ret, dtype=float), sector_excess])

    if intercept:
        X = np.column_stack([X, np.ones(len(X))])

    y_all = np.asarray(stock_ret, dtype=float)
    betas = np.full((len(y_all), 2), np.nan)
    resid_std = np.full(len(y_all), np.nan)

    for i in range(window, len(y_all)):
        y, Xi = y_all[i - window:i], X[i - window:i]
        mask = np.isfinite(y) & np.isfinite(Xi).all(axis=1)
        if mask.sum() > window // 2:
            coef = np.linalg.lstsq(Xi[mask], y[mask], rcond=None)[0]
            betas[i] = coef[:2]
            resid_std[i] = np.std(y[mask] - Xi[mask] @ coef, ddof=Xi.shape[1])

    return betas, resid_std


def rolling_two_factor_beta(stock_ret, spy_ret, sector_ret, window: int = 60):
    betas, _ = _rolling_two_factor_fit(stock_ret, spy_ret, sector_ret, window, intercept=False)
    return betas[:, 0], betas[:, 1]


def rolling_idiosyncratic_volatility(stock_ret, spy_ret, sector_ret, window: int = 60):
    """Annualised volatility of daily returns unexplained by the market and sector."""
    _, resid_std = _rolling_two_factor_fit(stock_ret, spy_ret, sector_ret, window, intercept=True)
    return resid_std * np.sqrt(252)
