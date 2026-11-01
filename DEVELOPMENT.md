<div align="center">
  <h2><b>Simple Alpha: Development </b></h2>
</div>

This is a guide for how to run run individual services. Running any tools using the forecasting api requires a valid checkpoint in `artifacts/checkpoints/best_mlflow_checkpoint.ckpt`. A default checkpoint is set in the artifacts directory so that the API can be run without any extra steps. It is however possible to use a custom checkpoint for the api but this is explained in a later section.

## Forecasting API

```bash
uv run uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```

**Test health**

```bash
http://127.0.0.1:8000/health
```

**Test Prediction**

```bash
curl -H 'Content-Type: application/json' \
      -d '{ "ticker":"AAPL","as_of_date":"2026-10-12"}' \
      -X POST \
      http://127.0.0.1:8000/predict
```

## MCP Server

The MCP server will start without the API running but none of the tools related to the API will work. Start the API in the background if you want to test these tools.

**1. Start the server**

```bash
uv sync
source .venv/bin/activate
python -m mcp_server.server
```

**2. Inspect the server using the MCP inspector**

```bash
npx @modelcontextprotocol/inspector
```

## LLM Harness
The harness can be run either as a standalone script or as a service which can communicate through websocket. The harness requires either a local LLM or an API key from a cloud provider which accepts either an OpenAI or Anthropic compatible chat completion endpoint. The harness also contains a number of tests which are mainly for testing tool use for smaller models, which can be inconsistent.

### Local LLM
The project was tested localy using [llama.cpp](https://github.com/ggml-org/llama.cpp) running against a Vulkan backend on windows. Tests were run with _Qwen3-14B-Q4_K_M.gguf_ and were stable for the most part but running a smaller model will degrade tool use. If you run a local model through llama.cpp then make sure it's running in server mode on port `8080` and host `0.0.0.0`.

### Cloud Model
Set the api key in the `.env` file and the corresponding provider will appear when running the harness against the frontend.

### Harness
The harness reads the addresses of the MCP server and the local LLM from `.env`. Both are required when running outside Docker (`docker compose` sets them itself):
```
MCP_URL=http://127.0.0.1:8001/mcp
LLM_URL=http://127.0.0.1:8080/v1
```

The harness can either be run so that it can be interacted with through the cmd or as a server to communicate with the frontend.

When run as a server, the WebSocket only accepts browser connections from the origins in `ALLOWED_ORIGINS` (default `http://localhost:5173,http://127.0.0.1:5173`). Set it in `.env` if you serve the frontend from another address.

**CMD interactable**
```bash
uv sync
source .venv/bin/activate
python -m llm_harness.harness              # local model
python -m llm_harness.harness --model anthropic:claude-sonnet-5-5
```

**As a service**
```bash
uv sync
source .venv/bin/activate
uvicorn llm_harness.ws_server:app --port 8002
```

**Tool-use tests**

With the MCP server running, grade how well one or more models pick tools and arguments. Models are the ids offered in the UI (`local`, `openai:<name>`, `anthropic:<name>`), or `all` for every model with an API key set:
```bash
python -m llm_harness.testing.test --models local anthropic:claude-sonnet-5-5
python -m llm_harness.testing.test test_cases_multi_prompt --models all
```
Grades are written to `llm_harness/testing/results.json` and per-case traces to `llm_harness/testing/traces/<model>/`; both are git-ignored.

## Frontend

```bash
uv sync
source .venv/bin/activate
cd frontend
npm run dev
```

## Set a custom checkpoint

Training logs runs and checkpoints to an MLflow server hosted on [DagsHub](https://dagshub.com/dashboard), so you need your own account and an (empty) repository there. Create a token at *Settings → Tokens*, then set `DAGSHUB_USER_TOKEN`, `DAGSHUB_REPO_OWNER` (your username) and `DAGSHUB_REPO_NAME` in `.env`, and follow the steps below. The training data is built locally in step 2; nothing needs to be pulled from DagsHub.

> **Warning:** checkpoints are Python pickle files and are loaded with `torch.load(..., weights_only=False)`, which can execute arbitrary code. Only load checkpoints you trained yourself or fully trust, and never point `CHECKPOINT_PATH` at a file from an untrusted source.

**1. Install the training dependencies**

```bash
uv sync --group training
```

**2. Build the training data**

The training data is not distributed with this repository. It is downloaded from Yahoo Finance (and optionally FRED and SEC EDGAR) into `data_pipeline/local_data/`, which is gitignored. See [Data and licensing](./README.md#data-and-licensing) before using it.

```bash
uv run python -m data_pipeline.fetch_data
```

This fetches daily prices for the S&P 500 constituents in `data_pipeline/tickers_sp500.yaml` and writes one feature file per ticker to `data_pipeline/local_data/raw_data`. The first run downloads the full history and takes a while; later runs only fetch what is new. Useful options: `--tickers AAPL MSFT` for a subset, `--workers <n>` for parallelism (keep it low to avoid being rate limited) and `--force` to redownload everything.

- `FRED_API_KEY` in `.env` (free, see `.env.example`) is optional. When it is set, the FRED macro features are added to the data. Without it they are skipped, so only configs that use those features need it.
- Fundamentals from SEC EDGAR are also optional and only needed if your config lists fundamentals features such as `gross_profitability` or `book_to_market`. Set `SEC_USER_AGENT` in `.env` to a descriptive name and email address (SEC requires it), then run `uv run python -m data_pipeline.edgar.build`. The output goes to `data_pipeline/local_data/fundamentals`.

Yahoo Finance data changes over time (revisions, delistings, constituents that have since left the index), so a freshly built dataset will not exactly match the one used for the published checkpoint. Expect results to differ slightly.

**3. Train**

Paths, `--config` included, are relative to the repo root. The run, with its best checkpoint, is logged to the MLflow server on DagsHub.

```bash
uv run python -m ML_layer.train_model --config ML_layer/model-configs/ptf-config.yaml
```

To score a trained run against the Persistence and Ridge baselines (set `MLFLOW_RUN_ID` in `.env`), use the same config. Its `test:` section is merged over the training settings, so the data and model always match the checkpoint:

```bash
uv run python -m ML_layer.evaluate --config ML_layer/model-configs/ptf-config.yaml [--permutation-importance]
```

**4. Pull the checkpoint and prepare it for the API**

This downloads the run's checkpoint to `artifacts/checkpoints/best_mlflow_checkpoint.ckpt` and embeds the model config and serving bundle the API needs. It reads the training data built in step 2. Use the run id shown in MLflow, or set `MLFLOW_RUN_ID` in `.env`:

```bash
uv run python -m ML_layer.prepare_checkpoint --run_id <mlflow-run-id>
```

Use the same config file that the run was trained with (`--config`, default `ML_layer/model-configs/ptf-config.yaml`) if the checkpoint has no embedded config. To prepare a checkpoint you already have locally, pass `--checkpoint <path>` instead of `--run_id`.
