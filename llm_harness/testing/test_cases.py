from .specs import exact, within

test_cases = {
  "tests": [
    {
      "id": "fc_001",
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
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_002",
      "prompts": [
        {
          "prompt": "What's the price history for AAPL?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": within(20, 60)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_003",
      "prompts": [
        {
          "prompt": "What is the forecast for MSFT?",
          "expected_tool_calls": [
            {"name": "forecast", "args": {"ticker": exact("MSFT")}},
            {"name": "forecast_reasoning", "args": {"ticker": exact("MSFT")}},
            {"name": "price_history", "args": {"ticker": exact("MSFT"), "days": within(20, 60)}},
            {"name": "find_news_articles", "args": {"ticker": exact("MSFT"), "count": within(1, 5)}}
          ],
          "category": "multi_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_004",
      "prompts": [
        {
          "prompt": "Show a forecast chart for NVDA as of 2024-06-30.",
          "expected_tool_calls": [
            {"name": "forecast_plot", "args": {"ticker": exact("NVDA"), "as_of_date": exact("2024-06-30")}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_005",
      "prompts": [
        {
          "prompt": "Give me the reasoning behind the Tesla forecast.",
          "expected_tool_calls": [
            {"name": "forecast_reasoning", "args": {"ticker": exact("TSLA")}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_006",
      "prompts": [
        {
          "prompt": "How has GOOGL's price changed over the past 60 days?",
          "expected_tool_calls": [
            {"name": "price_history", "args": {"ticker": exact("GOOGL"), "days": exact(60)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_007",
      "prompts": [
        {
          "prompt": "Plot Amazon's price history for the last 90 days.",
          "expected_tool_calls": [
            {"name": "plot_price_history", "args": {"ticker": exact("AMZN"), "days": exact(90)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_008",
      "prompts": [
        {
          "prompt": "What is the latest news about the google stock?",
          "expected_tool_calls": [
            {"name": "find_news_articles", "args": {"ticker": exact("GOOG"), "count": within(1, 5)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_009",
      "prompts": [
        {
          "prompt": "Give me the latest 5 articles about Google stock.",
          "expected_tool_calls": [
            {"name": "find_news_articles", "args": {"ticker": exact("GOOGL"), "count": exact(5)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_010",
      "prompts": [
        {
          "prompt": "I need the Netflix forecast for 2024-04-15.",
          "expected_tool_calls": [
            {"name": "forecast", "args": {"ticker": exact("NFLX"), "as_of_date": exact("2024-04-15")}},
            {"name": "forecast_reasoning", "args": {"ticker": exact("NFLX"), "as_of_date": exact("2024-04-15")}},
            {"name": "price_history", "args": {"ticker": exact("NFLX"), "days": within(20, 60)}}
          ],
          "category": "multi_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_011",
      "prompts": [
        {
          "prompt": "Check if the forecast service is healthy before I trust it.",
          "expected_tool_calls": [
            {"name": "health_check", "args": {}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_012",
      "prompts": [
        {
          "prompt": "What is the forecast for AAPL on 2024-01-31 and compare it to recent price history?",
          "expected_tool_calls": [
            {"name": "forecast", "args": {"ticker": exact("AAPL"), "as_of_date": exact("2024-01-31")}},
            {"name": "price_history", "args": {"ticker": exact("AAPL"), "days": within(20, 60)}}
          ],
          "category": "multi_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_013",
      "prompts": [
        {
          "prompt": "Summarize the recent price action for IBM and show a chart.",
          "expected_tool_calls": [
            {"name": "plot_price_history", "args": {"ticker": exact("IBM"), "days": within(20, 60)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_014",
      "prompts": [
        {
          "prompt": "Tell me the latest market headlines for NVDA.",
          "expected_tool_calls": [
            {"name": "find_news_articles", "args": {"ticker": exact("NVDA"), "count": within(1, 5)}}
          ],
          "category": "single_tool"
        }
      ],
      "ideal_attempts": 2,
      "multi_prompt": False
    },
    {
      "id": "fc_015",
      "prompts": [
        {
          "prompt": "How's the weather in Seattle today?",
          "expected_tool_calls": [],
          "category": "no_tool"
        }
      ],
      "ideal_attempts": 1,
      "multi_prompt": False
    }
  ]
}