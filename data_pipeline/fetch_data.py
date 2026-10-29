"""Fetch daily OHLCV data and build leakage-safe model features: the ``python -m data_pipeline.fetch_data`` entry point."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .utils.cache import load_or_refresh_earnings, load_or_refresh_market_context, source_cache_path
from .utils.config import DEFAULT_TICKER_FILE, FEATURE_WARMUP_YEARS, HISTORY_START_TOLERANCE, history_start, load_tickers
from .utils.features import engineer_features
from .utils.fred import fetch_fred_context
from .yahoo import SECTOR_NAME_MAPPINGS, fetch_ohlcv, fetch_sector, normalise_index


def save_raw_features(features: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    raw = features.copy()
    raw.index.name = "date"
    raw = raw.reset_index()
    raw.insert(0, "ticker", ticker)
    out_path = output_dir / f"{ticker}_raw.parquet"
    raw.to_parquet(out_path, index=False)

    return out_path


def process_ticker(
    ticker: str,
    mkt: pd.DataFrame,
    output_dir: Path,
    fred_context: pd.DataFrame | None = None,
    force: bool = False,
) -> tuple[str, int]:
    """Refresh one ticker from its cached last date and rebuild its features.

    The source OHLCV sidecar is required because returns, rolling volume, RSI,
    and beta need prior observations.  Rebuilding feature values locally is
    cheap and avoids a boundary gap while the network request remains strictly
    incremental.
    """

    source_path = source_cache_path(ticker, output_dir)
    cached_source = pd.read_parquet(source_path) if source_path.exists() else None

    # Sectors rarely change, so reuse a valid cached one (even with --force)
    # instead of hitting Yahoo's heavily rate-limited quote-summary endpoint.
    cached_sector = None
    if cached_source is not None and "sector" in cached_source:
        cached_sector = cached_source["sector"].iloc[-1]
    sector = cached_sector if cached_sector in SECTOR_NAME_MAPPINGS else None

    # Sidecars record the start they were downloaded from, so a longer
    # FEATURE_START triggers one full redownload, while tickers listed after
    # that start are not refetched on every run.
    if cached_source is not None:
        requested = cached_source["history_start"].iloc[-1] if "history_start" in cached_source else None
        if requested is None or pd.Timestamp(requested) > history_start(FEATURE_WARMUP_YEARS) + HISTORY_START_TOLERANCE:
            cached_source = None

    if force:
        cached_source = None

    if cached_source is None:
        # Pre-sidecar feature files cannot safely reconstruct historical volume,
        # high/low, or RSI state.  Do one migration download, then all later
        # runs are incremental.
        requested_start = history_start(FEATURE_WARMUP_YEARS)
        df, sector = fetch_ohlcv(ticker, start=requested_start, sector=sector)

    else:
        requested_start = pd.Timestamp(cached_source["history_start"].iloc[-1])
        cached_source = normalise_index(cached_source.set_index("date"))
        sector = sector or fetch_sector(ticker)
        try:
            fresh, _ = fetch_ohlcv(ticker, start=cached_source.index.max(), sector=sector)
        except ValueError as exc:
            if "No OHLCV rows returned" not in str(exc):
                raise
            fresh = cached_source.iloc[:0].drop(columns=["sector", "history_start"], errors="ignore")
        df = pd.concat([cached_source.drop(columns=["sector", "history_start"], errors="ignore"), fresh]).loc[
            lambda x: ~x.index.duplicated(keep="last")
        ].sort_index()

    # Dates are in yfinance's index; an index join prevents the prior Date-column
    # merge bug and validates that context cannot duplicate ticker rows.
    df = df.join(mkt, how="left", validate="one_to_one")
    if fred_context is not None and not fred_context.empty:
        df = df.join(fred_context, how="left", validate="one_to_one")


    earnings = load_or_refresh_earnings(ticker, output_dir, force)
    features = engineer_features(df, sector, earnings=earnings)

    # Drop the warm-up year used only to initialise rolling features.
    features = features[features.index >= history_start()]
    save_raw_features(features, ticker, output_dir)

    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_columns = ["Open", "High", "Low", "Close", "Volume"]
    source = df[source_columns].reset_index(names="date")
    source["sector"] = sector
    source["history_start"] = requested_start
    source.to_parquet(source_path, index=False)

    return ticker, len(features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--ticker_file", default=str(DEFAULT_TICKER_FILE))
    parser.add_argument("--output_dir", default=str(Path(__file__).parent / "local_data" / "raw_data"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fred-api-key", default=None)
    parser.add_argument("--force", action="store_true", help="Redownload complete ticker and market histories")

    args = parser.parse_args()

    tickers, indexes = load_tickers(Path(args.ticker_file))

    if args.tickers:
        tickers = args.tickers

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    load_dotenv()

    print("Refreshing market context...")
    mkt = load_or_refresh_market_context(indexes, output_dir, args.force)

    print("Fetching FRED context...")
    fred_context = fetch_fred_context(mkt.index, args.fred_api_key, include_levels=True)

    if fred_context.empty:
        print("Unable to fetch FRED context")

    results, errors = [], []
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_ticker, t, mkt, output_dir, fred_context, args.force): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, rows = future.result()
                print(f"  ✓ {ticker}: {rows} rows")
                results.append(ticker)
            except Exception as exc:
                print(f"  ✗ {ticker}: {exc}")
                errors.append(ticker)
    print(f"Done: {len(results)} succeeded, {len(errors)} failed.")
    if errors:
        print(f"Failed tickers: {errors}")


if __name__ == "__main__":
    main()
