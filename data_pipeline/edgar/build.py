"""Build the point-in-time EDGAR fundamentals features for a ticker universe.

Usage: python -m data_pipeline.edgar.build [--ticker-file data_pipeline/tickers_sp500.yaml] [--tickers AAPL JPM]

Needs SEC_USER_AGENT (see client.py). Raw SEC downloads and parsed tables are cached under
``local_data/edgar_cache`` so reruns do not download again; features are written to
``local_data/fundamentals/<ticker>.parquet`` (one row per trading date of the ticker's raw feature file).
"""

import argparse
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from ..utils.config import load_tickers
from ..yahoo import YAHOO_ATTEMPTS, yahoo_backoff
from .client import SecClient, read_company_facts
from .facts import CHAINS, PERIODIC_FORMS, parse_company_facts
from .features import (FINANCIAL_SIC, FUNDAMENTAL_FEATURES, build_daily_features, compute_filing_events,
                       earnings_announcement_returns, earnings_announcements)

LOCAL_DATA = Path(__file__).resolve().parents[1] / "local_data"
PRICE_START = "2008-01-01"
EXTREME_ROWS = 20


def parse_submissions(pages: list[dict]) -> tuple[int | None, pd.DataFrame]:
    """SIC code and every filing (form, filed, items) across the recent and older-filings pages."""
    sic = pages[0].get("sic")
    frames = []
    for page in pages:
        recent = page.get("filings", {}).get("recent", page)
        if "form" in recent:
            frames.append(pd.DataFrame({"form": recent["form"], "filed": pd.to_datetime(recent["filingDate"]),
                                        "items": recent.get("items", [""] * len(recent["form"]))}))
    filings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["form", "filed", "items"])
    return (int(sic) if sic else None), filings


def fetch_unadjusted_prices(ticker: str, cache_directory: Path) -> pd.DataFrame:
    """Split-adjusted, dividend-unadjusted close and split ratios (0 when none), cached."""
    path = cache_directory / "prices" / f"{ticker}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    for attempt in range(YAHOO_ATTEMPTS):
        data = yf.download(ticker, start=PRICE_START, auto_adjust=False, actions=True, progress=False)
        if not data.empty:
            break
        if attempt < YAHOO_ATTEMPTS - 1:
            yahoo_backoff(attempt)
    if data.empty:
        raise ValueError(f"No unadjusted prices for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    prices = pd.DataFrame({"close": data["Close"], "split": data.get("Stock Splits", 0.0)}).fillna({"split": 0.0})
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices


def process_ticker(ticker: str, cik: int, raw_directory: Path, output_directory: Path, cache_directory: Path) -> dict:
    """Features for one ticker. Returns bookkeeping for the coverage report."""
    facts_path = cache_directory / "facts" / f"CIK{cik:010d}.parquet"
    if facts_path.exists():
        facts = pd.read_parquet(facts_path)
    else:
        payload = read_company_facts(cache_directory / "companyfacts.zip", cik)
        if payload is None:
            return {"ticker": ticker, "error": "no company facts"}
        facts = parse_company_facts(payload)
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        facts.to_parquet(facts_path)
    if facts.empty:
        return {"ticker": ticker, "error": "no facts for the tag chains"}
    sic, filings = parse_submissions_cached(cache_directory, cik)
    raw = pd.read_parquet(raw_directory / f"{ticker}_raw.parquet", columns=["date", "target_log_close", "target_spy_log_close"])
    raw = raw.sort_values("date").set_index("date")
    prices = fetch_unadjusted_prices(ticker, cache_directory)
    splits = prices.loc[prices["split"] > 0, "split"]

    financial = sic is not None and sic in FINANCIAL_SIC
    events, flagged = compute_filing_events(facts, splits, financial)
    periodic = filings[filings["form"].isin(PERIODIC_FORMS)]
    if periodic.empty:  # fall back to the facts' own filing dates when the submissions list is unavailable
        periodic = facts[["form", "filed"]].drop_duplicates()
    announcements = earnings_announcements(periodic, filings[filings["form"].isin(["8-K", "8-K/A"])])
    ear = earnings_announcement_returns(announcements, raw["target_log_close"], raw["target_spy_log_close"])
    daily = build_daily_features(raw.index, events, ear, prices["close"])
    output_directory.mkdir(parents=True, exist_ok=True)
    daily.rename_axis("date").reset_index().to_parquet(output_directory / f"{ticker}.parquet", index=False)

    tag_columns = [f"tag_{item}" for item in CHAINS]
    tags = events.assign(year=events["filed"].dt.year)[["year", *tag_columns]].melt("year", var_name="item", value_name="tag").dropna()
    tags["item"] = tags["item"].str.removeprefix("tag_")
    return {
        "ticker": ticker, "sic": sic, "financial": financial, "first_filing": events["filed"].min(),
        "tag_usage": Counter(map(tuple, tags[["year", "item", "tag"]].itertuples(index=False, name=None))),
        "flagged_share_jumps": [{"ticker": ticker, **f} for f in flagged],
        "ear_sources": Counter(zip(pd.to_datetime(ear["announced"]).dt.year, np.where(ear["fallback"], "fallback", "8-K 2.02"))),
        "rows": len(daily),
    }


def parse_submissions_cached(cache_directory: Path, cik: int) -> tuple[int | None, pd.DataFrame]:
    directory = cache_directory / "submissions_parsed"
    path = directory / f"CIK{cik:010d}.parquet"
    if not path.exists():
        return None, pd.DataFrame(columns=["form", "filed", "items"])
    sic = (directory / f"CIK{cik:010d}.sic").read_text()
    return (int(sic) if sic else None), pd.read_parquet(path)


def fetch_submissions(client: SecClient, cik: int, cache_directory: Path) -> None:
    directory = cache_directory / "submissions_parsed"
    path = directory / f"CIK{cik:010d}.parquet"
    if path.exists():
        return
    sic, filings = parse_submissions(client.submissions(cik))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"CIK{cik:010d}.sic").write_text("" if sic is None else str(sic))
    filings.to_parquet(path, index=False)


def write_coverage_report(output_directory: Path, raw_directory: Path, results: list[dict], unmapped: list[str],
                          failed: dict[str, str], path: Path) -> None:
    ok = [r for r in results if "error" not in r]
    frames = [pd.read_parquet(output_directory / f"{r['ticker']}.parquet").assign(ticker=r["ticker"]) for r in ok]
    data = pd.concat(frames, ignore_index=True)
    data["year"] = data["date"].dt.year
    lines = ["# Fundamentals coverage", "",
             f"- Tickers with features: {len(ok)}; unmapped to a CIK: {len(unmapped)}; failed: {len(failed)}",
             f"- Window: {data['date'].min().date()} to {data['date'].max().date()}", ""]
    coverage = data.groupby("year")[FUNDAMENTAL_FEATURES].apply(lambda g: g.notna().mean() * 100).round(1)
    lines += ["## Per-year coverage (% of ticker-days with a value)", "", coverage.to_markdown(), ""]
    usage = Counter()
    for r in ok:
        usage.update(r["tag_usage"])
    if usage:
        table = pd.Series(usage).rename_axis(["year", "item", "tag"]).rename("filings").reset_index()
        lines += ["## Tag usage per year (filings whose latest value came from each tag)", ""]
        for item, g in table.groupby("item"):
            lines += [f"### {item}", "", g.pivot_table(index="year", columns="tag", values="filings", fill_value=0).to_markdown(), ""]
    sources = Counter()
    for r in ok:
        sources.update(r["ear_sources"])
    if sources:
        s = pd.Series(sources).rename_axis(["year", "source"]).rename("announcements").reset_index()
        pivot = s.pivot_table(index="year", columns="source", values="announcements", fill_value=0)
        pivot["fallback_share_%"] = (pivot.get("fallback", 0) / pivot.sum(axis=1) * 100).round(1)
        total = s.groupby("source")["announcements"].sum()
        lines += ["## Earnings announcement date source", "", f"Overall: {(total / total.sum() * 100).round(1).to_dict()}", "", pivot.to_markdown(), ""]
    lines += ["## Unmapped tickers (no CIK in SEC company_tickers.json)", "", ", ".join(unmapped) or "none", ""]
    if failed:
        lines += ["## Failed tickers", ""] + [f"- {t}: {e}" for t, e in failed.items()] + [""]
    jumps = pd.DataFrame([j for r in ok for j in r["flagged_share_jumps"]])
    lines += ["## Flagged share-count jumps (|log change| > 1.5; feature set to NaN for review)", ""]
    lines += [jumps.to_markdown(index=False) if len(jumps) else "none", ""]
    lines += [f"## {EXTREME_ROWS} most extreme values per feature (by |value|), for manual spot checks", ""]
    for feature in FUNDAMENTAL_FEATURES:
        top = data.dropna(subset=[feature]).loc[lambda d: d[feature].abs().nlargest(EXTREME_ROWS).index, ["ticker", "date", feature]]
        lines += [f"### {feature}", "", top.assign(date=top["date"].dt.date).to_markdown(index=False), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def spot_check(output_directory: Path, tickers: list[str]) -> None:
    """Print each ticker's feature value at the start of each year-end window for manual review."""
    for ticker in tickers:
        path = output_directory / f"{ticker}.parquet"
        if not path.exists():
            print(f"{ticker}: no fundamentals file")
            continue
        frame = pd.read_parquet(path).set_index("date")
        changes = frame[(frame != frame.shift()).any(axis=1)]
        print(f"\n== {ticker}: {len(frame)} days; feature values at each change (last 12) ==")
        print(changes.tail(12).round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ticker-file", default=str(Path(__file__).resolve().parents[1] / "tickers_sp500.yaml"))
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--raw-directory", default=str(LOCAL_DATA / "raw_data"))
    parser.add_argument("--output-directory", default=str(LOCAL_DATA / "fundamentals"))
    parser.add_argument("--cache-directory", default=str(LOCAL_DATA / "edgar_cache"))
    parser.add_argument("--report", default=str(Path(__file__).resolve().parents[2] / "outputs/fundamentals/fundamentals_coverage.md"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--spot-check", nargs="*", default=["AAPL", "JPM", "XOM"])
    args = parser.parse_args()
    load_dotenv()
    tickers = args.tickers or load_tickers(Path(args.ticker_file))[0]
    raw_directory, output_directory, cache_directory = Path(args.raw_directory), Path(args.output_directory), Path(args.cache_directory)
    tickers = [t for t in tickers if (raw_directory / f"{t}_raw.parquet").exists()]

    client = SecClient(cache_directory)
    cik_by_ticker = client.ticker_to_cik()
    unmapped = [t for t in tickers if t.upper() not in cik_by_ticker]
    mapped = {t: cik_by_ticker[t.upper()] for t in tickers if t.upper() in cik_by_ticker}
    print(f"{len(mapped)} tickers mapped to a CIK; unmapped ({len(unmapped)}): {unmapped}", flush=True)

    print("Downloading company facts (cached after the first run)...", flush=True)
    client.company_facts_zip()
    print("Fetching submissions...", flush=True)
    for cik in sorted(set(mapped.values())):
        fetch_submissions(client, cik, cache_directory)
    print("Fetching unadjusted prices...", flush=True)
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch_unadjusted_prices, t, cache_directory): t for t in mapped}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failed[futures[future]] = str(exc)
    mapped = {t: c for t, c in mapped.items() if t not in failed}

    print("Building features...", flush=True)
    results = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_ticker, t, c, raw_directory, output_directory, cache_directory): t for t, c in mapped.items()}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failed[ticker] = repr(exc)
                continue
            if "error" in result:
                failed[ticker] = result["error"]
            else:
                results.append(result)
    print(f"{len(results)} tickers built in {time.time() - start:.0f}s; {len(failed)} failed", flush=True)
    write_coverage_report(output_directory, raw_directory, results, unmapped, failed, Path(args.report))
    print(f"Coverage report: {args.report}")
    spot_check(output_directory, args.spot_check)


if __name__ == "__main__":
    main()
