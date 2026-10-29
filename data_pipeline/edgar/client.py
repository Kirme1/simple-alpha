"""SEC EDGAR access: identified, rate-limited, retried and cached downloads.

SEC fair-access rules: every request carries a User-Agent with contact details
(env var SEC_USER_AGENT) and stays at or below 10 requests/second; 8 is used here.
"""

import json
import os
import threading
import time
import zipfile
from pathlib import Path

import requests

USER_AGENT_ENV = "SEC_USER_AGENT"
MAX_REQUESTS_PER_SECOND = 8
ATTEMPTS = 5
TIMEOUT_SECONDS = 120

COMPANY_FACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"


def user_agent() -> str:
    agent = os.environ.get(USER_AGENT_ENV, "").strip()
    if not agent:
        raise RuntimeError(
            f"{USER_AGENT_ENV} is not set. The SEC requires contact details in the User-Agent; "
            f"set it to e.g. 'Your Name your.email@example.com' (in .env or the environment)."
        )
    return agent


class SecClient:
    def __init__(self, cache_directory: Path):
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent()
        self._lock = threading.Lock()
        self._next_request = 0.0

    def _wait_for_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._next_request = max(self._next_request, now) + 1.0 / MAX_REQUESTS_PER_SECOND
            delay = self._next_request - 1.0 / MAX_REQUESTS_PER_SECOND - now
        if delay > 0:
            time.sleep(delay)

    def get(self, url: str, stream: bool = False) -> requests.Response:
        error: Exception | None = None
        for attempt in range(ATTEMPTS):
            self._wait_for_slot()
            try:
                response = self.session.get(url, stream=stream, timeout=TIMEOUT_SECONDS)
                if response.status_code == 200:
                    return response
                if response.status_code not in (403, 429, 500, 502, 503, 504):
                    response.raise_for_status()
                error = RuntimeError(f"HTTP {response.status_code} for {url}")
            except requests.RequestException as exc:
                error = exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"SEC request failed after {ATTEMPTS} attempts: {url}: {error}")

    def download(self, url: str, path: Path) -> Path:
        """Stream a large file to ``path`` once; later calls reuse it."""
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        with self.get(url, stream=True) as response, open(partial, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        partial.rename(path)
        return path

    def get_json(self, url: str, path: Path) -> dict:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.get(url).text)
        return json.loads(path.read_text())

    def ticker_to_cik(self) -> dict[str, int]:
        raw = self.get_json(COMPANY_TICKERS_URL, self.cache_directory / "company_tickers.json")
        return {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}

    def company_facts_zip(self) -> Path:
        return self.download(COMPANY_FACTS_URL, self.cache_directory / "companyfacts.zip")

    def submissions(self, cik: int) -> list[dict]:
        """The submissions document plus every older-filings page it references."""
        directory = self.cache_directory / "submissions"
        main = self.get_json(SUBMISSIONS_URL.format(cik=cik), directory / f"CIK{cik:010d}.json")
        pages = [main]
        for extra in main.get("filings", {}).get("files", []):
            pages.append(self.get_json(SUBMISSIONS_PAGE_URL.format(name=extra["name"]), directory / extra["name"]))
        return pages


def read_company_facts(zip_path: Path, cik: int) -> dict | None:
    with zipfile.ZipFile(zip_path) as archive:
        name = f"CIK{cik:010d}.json"
        if name not in archive.namelist():
            return None
        with archive.open(name) as f:
            return json.load(f)
