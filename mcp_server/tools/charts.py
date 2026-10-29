"""Chart payloads sent to the frontend alongside a tool's text summary.

A chart travels as an embedded JSON resource. The harness forwards it to the browser as a
`chart` event and keeps it out of the LLM's context (the LLM only receives the text blocks).
Keep CHART_MIME in sync with llm_harness/harness.py.
"""

import json

from mcp.types import EmbeddedResource, TextResourceContents

CHART_MIME = "application/vnd.simple-alpha.chart+json"


def chart_block(kind: str, ticker: str, **data) -> EmbeddedResource:
    payload = {"kind": kind, "ticker": ticker, **data}
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=f"chart://{kind}/{ticker}",
            mimeType=CHART_MIME,
            text=json.dumps(payload),
        ),
    )
