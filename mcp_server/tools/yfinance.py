"""client used by the MCP service to call the yfinance API."""

from mcp.types import TextContent, EmbeddedResource
import yfinance as yf
import pandas as pd
import httpx
from urllib.parse import urlsplit
from bs4 import BeautifulSoup

from .charts import chart_block
from .config import Settings
from .responseSchema import *


_ALLOWED_ARTICLE_HOST = "finance.yahoo.com"
_MAX_REDIRECTS = 5


def _is_allowed_article_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" and parts.hostname == _ALLOWED_ARTICLE_HOST and parts.port is None


class YfinanceTools:
    def __init__(self, settings: Settings) -> None:
        self._timeout = settings.forecast_api_timeout_seconds

    async def get_price_history(
        self, ticker: str, days: int
    ) -> PriceHistory | list[TextContent]:
        """Return the price history for a given ticker and number of days."""
        try:
            ticker = ticker.upper().strip()
            data = yf.Ticker(ticker).history(period=f"{days}d", interval="1d", timeout=self._timeout)

            if data.empty:
                return [
                    TextContent(type="text", text=f"Incomplete data for ticker {ticker}. It may be an invalid ticker symbol.")
                ]

            if len(data) < days * 0.5:  # Allow for some missing days due to holidays, etc.
                return [
                    TextContent(type="text", text=f"Incomplete data for ticker {ticker}. Only {len(data)} trading days available out of the requested {days} days.")
                ]
            

            return PriceHistory(
                Ticker=ticker,
                Trading_Days=len(data),
                Start_Date=data.index[0].date(),
                End_Date=data.index[-1].date(),
                Start_Close=round(float(data['Close'].iloc[0]), 2),
                End_Close=round(float(data['Close'].iloc[-1]), 2),
                period_return_pct=round(float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100), 2),
                max_drawdown_pct=round(float(((data['Close'] / data['Close'].cummax()) - 1).min() * 100), 2),
                avg_daily_volume=round(float(data['Volume'].mean()))
            )

        except Exception as e:
            return [
                TextContent(type="text", text=f"Error fetching price history for ticker {ticker}: {str(e)}")
            ]

    async def get_market_news(
            self, ticker: str, count: int
        ) -> MarketNews |list[TextContent]:
            """Return the latest market news for a given ticker."""
            ticker = ticker.upper().strip()
            extracted_news = []
            try:
                # Search can return broader articles that merely mention the query,
                # so over-fetch and keep only those Yahoo tags with this ticker.
                news = yf.Search(ticker, news_count=count * 2, max_results=0, timeout=self._timeout).news
                news = [a for a in news if ticker in (a.get("relatedTickers") or [])][:count]

                if len(news) < count:
                    return [
                        TextContent(type="text", text=f"Incomplete data for ticker {ticker}. Only {len(news)} news items available out of the requested {count}.")
                    ]

                for article in news:
                    extracted_news.append({
                        "title": article['title'],
                        "provider": article['publisher'],
                        "link": article['link'],
                        "pubDate": pd.Timestamp(article['providerPublishTime'], unit='s', tz='UTC').isoformat(),
                    })

                return MarketNews(
                    Ticker=ticker,
                    News_Items=extracted_news
                )

            except Exception as e:
                return [
                    TextContent(type="text", text=f"Error fetching market news for ticker {ticker}: {str(e)}")
                ]

    async def get_article_content(
        self, url: str, max_chars: int
    ) -> ArticleContent | list[TextContent]:
        """Return the text of a Yahoo Finance news article."""
        try:
            # The URL comes from the LLM, so only Yahoo Finance pages may be fetched;
            # redirects are followed manually so every hop is checked as well.
            async with httpx.AsyncClient(timeout=self._timeout, headers={"User-Agent": "Mozilla/5.0"}) as client:
                for _ in range(_MAX_REDIRECTS + 1):
                    if not _is_allowed_article_url(url):
                        return [
                            TextContent(type="text", text=f"Refusing to fetch {url}: only https://{_ALLOWED_ARTICLE_HOST} links from market_news are supported.")
                        ]
                    response = await client.get(url)
                    if not response.is_redirect:
                        break
                    url = str(response.url.join(response.headers["location"]))
                else:
                    return [TextContent(type="text", text=f"Too many redirects fetching {url}.")]

            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("article")
            paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")] if article else []
            content = "\n\n".join(p for p in paragraphs if p)

            if not content:
                return [
                    TextContent(type="text", text=f"Could not extract article text from {url}. It may be a premium article or a link to an external site.")
                ]

            # The page's first <h1> is the site logo, so take the article's own heading.
            title = article.find("h1")
            return ArticleContent(
                Url=url,
                Title=title.get_text(" ", strip=True) if title else "",
                Content=content[:max_chars],
                Truncated=len(content) > max_chars,
            )

        except Exception as e:
            return [
                TextContent(type="text", text=f"Error fetching article {url}: {str(e)}")
            ]

    async def plot_price_history(
        self, ticker: str, days: int
    ) -> list[TextContent | EmbeddedResource]:
        """Return the price history plot for a given ticker and number of days."""
        try:
            ticker = ticker.upper().strip()
            data = yf.Ticker(ticker).history(period=f"{days}d", interval="1d", timeout=self._timeout)

            if data.empty:
                return [
                    TextContent(type="text", text=f"Incomplete data for ticker {ticker}. It may be an invalid ticker symbol.")
                ]

            if len(data) < days * 0.5:  # Allow for some missing days due to holidays, etc.
                return [
                    TextContent(type="text", text=f"Incomplete data for ticker {ticker}. Only {len(data)} trading days available out of the requested {days} days.")
                ]

            data = data.dropna(subset=["Open", "High", "Low", "Close"])
            candles = [
                {
                    "time": day.strftime("%Y-%m-%d"),
                    "open": round(float(row.Open), 2),
                    "high": round(float(row.High), 2),
                    "low": round(float(row.Low), 2),
                    "close": round(float(row.Close), 2),
                    "volume": int(row.Volume) if pd.notna(row.Volume) else 0,
                }
                for day, row in zip(data.index, data.itertuples())
            ]

            summary = (
                f"{ticker}: min {round(float(data['Close'].min()), 2)}, max {round(float(data['Close'].max()), 2)}. "
                "A candlestick chart is displayed to the user."
            )

            return [
                TextContent(type="text", text=summary),
                chart_block("price", ticker, candles=candles),
            ]

        except Exception as e:
            return [
                TextContent(type="text", text=f"Error fetching price history for ticker {ticker}: {str(e)}")
            ]

