from .specs import exact, within

test_cases = {
  "tests": [
    {
      "id": "fc_016_implicit_reference",
      "prompts": [
        {
          "prompt": "What's the forecast for AAPL?",
          "expected_tool_calls": [
            {"name": "forecast", "args": {"ticker": exact("AAPL")}},
            {"name": "forecast_reasoning", "args": {"ticker": exact("AAPL")}},
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": within(20, 60)}},
            {"name": "find_news_articles", "args": {"ticker": exact("AAPL"), "count": within(1, 5)}}
          ],
          "category": "multi_tool"
        },
        {
          "prompt": "What about AMZN?",
          "expected_tool_calls": [
            {"name": "forecast", "args": {"ticker": exact("AMZN")}},
            {"name": "forecast_reasoning", "args": {"ticker": exact("AMZN")}},
            {"name": "price_history", "args": {"ticker": exact("AMZN"), "days": within(20, 60)}},
            {"name": "find_news_articles", "args": {"ticker": exact("AMZN"), "count": within(1, 5)}}
          ],
          "category": "multi_tool"
        },
      ],
      "ideal_attempts": 3,
      "multi_prompt": True
    },
    {
      "id": "fc_017_correction_turn",
      "prompts": [
        {
          "prompt": "What's the price history for AAPL over the last 30 days?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": exact(30)}}
          ],
          "category": "single_tool"
        },
        {
          "prompt": "Actually, make that 15 days.",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": exact(15)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": True
    },
    {
      "id": "fc_018_distraction_turn",
      "prompts": [
        {
          "prompt": "Give me the forecast for NVDA.",
          "expected_tool_calls": [
            {"name": "forecast", "args": {"ticker": exact("NVDA")}},
            {"name": "forecast_reasoning", "args": {"ticker": exact("NVDA")}},
            {"name": "price_history", "args": {"ticker": exact("NVDA"), "days": within(20, 60)}},
            {"name": "find_news_articles", "args": {"ticker": exact("NVDA"), "count": within(1, 5)}}
          ],
          "category": "multi_tool"
        },
        {
          "prompt": "By the way, how's the weather in Seattle today?",
          "expected_tool_calls": [],
          "category": "no_tool"
        },
        {
          "prompt": "Now give me the forecast for NVDA again.",
          "expected_tool_calls": [
            {"name": "forecast", "args": {"ticker": exact("NVDA")}},
            {"name": "forecast_reasoning", "args": {"ticker": exact("NVDA")}},
            {"name": "price_history", "args": {"ticker": exact("NVDA"), "days": within(20, 60)}},
            {"name": "find_news_articles", "args": {"ticker": exact("NVDA"), "count": within(1, 5)}}
          ],
          "category": "multi_tool"
        }
      ],
      "ideal_attempts": 3,
      "multi_prompt": True
    },
    {
      "id": "fc_019_long_horizon_degradation",
      "prompts": [
        {
          "prompt": "What's the price history for AAPL over the last 30 days?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": exact(30)}}
          ],
          "category": "single_tool"
        },
        {
          "prompt": "What's the price history for AAPL over the last 30 days?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": exact(30)}}
          ],
          "category": "single_tool"
        },
        {
          "prompt": "What's the price history for AAPL over the last 30 days?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": exact(30)}}
          ],
          "category": "single_tool"
        },
        {
          "prompt": "What's the price history for AAPL over the last 30 days?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": exact(30)}}
          ],
          "category": "single_tool"
        },
        {
          "prompt": "What's the price history for AAPL over the last 30 days?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": exact(30)}}
          ],
          "category": "single_tool"
        },
        {
          "prompt": "What's the price history for AAPL over the last 30 days?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": exact(30)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": True
    },
    {
      "id": "fc_020_implicit_reference",
      "prompts": [
      {
        "prompt": "What's the forecast for AAPL?",
        "expected_tool_calls": [
          {"name": "forecast", "args": {"ticker": exact("AAPL")}},
          {"name": "forecast_reasoning", "args": {"ticker": exact("AAPL")}},
          {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": within(20, 60)}},
          {"name": "find_news_articles", "args": {"ticker": exact("AAPL"), "count": within(1, 5)}}
        ],
        "category": "multi_tool"
      },
      {
        "prompt": "What about AMZN?",
        "expected_tool_calls": [
          {"name": "forecast", "args": {"ticker": exact("AMZN")}},
          {"name": "forecast_reasoning", "args": {"ticker": exact("AMZN")}},
          {"name": "price_history", "args": {"ticker": exact("AMZN"), "days": within(20, 60)}},
          {"name": "find_news_articles", "args": {"ticker": exact("AMZN"), "count": within(1, 5)}}
        ],
        "category": "multi_tool"
      },
    ],
    "ideal_attempts": 3,
    "multi_prompt": True
    },
  ]
}