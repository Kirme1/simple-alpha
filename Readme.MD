<div align="center">
  <h2><b>Simple Alpha </b></h2>
</div>

> Simple Alpha is a small harness connected to a temporal fusion transformer which allows llms to forecast price movements for financial assets with comprehensible explainations

https://github.com/user-attachments/assets/a23510be-2f53-43b8-9719-7f87cc31328e

## Quickstart

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Step 1: clone repository**

```bash
git clone https://github.com/Kirme1/simple-alpha
cd SimpleAlpha
cp .env.example .env # add your API key here
```

**Step 2: Set an LLM**

- Option 1: If you have a API key from a provider which supports either an OpenAI or Anthropic chat completion endpoint then you can add it to the `.env` file.

- Option 2: If you want to run a local model instead then make sure it's running in server mode on port `8080` and host `0.0.0.0`. The harness will use this by default.

**Step 3: build**

```bash
docker compose up --build
```

**Step 4: open the app**

``` bash
http://localhost:5173/
```

The chat socket only accepts connections from `http://localhost:5173` and `http://127.0.0.1:5173`, so that other websites cannot use your API keys. If you open the app from another address, add it to `ALLOWED_ORIGINS` in `.env` (comma separated). Don't expose the app to a network you don't trust, as it has no authentication.

For running individual services during development, see [DEVELOPMENT.md](./DEVELOPMENT.md).

## Model Training

The model trained for predicting price movements is a [temporal fusion transformer](https://arxiv.org/abs/1912.09363) and uses the pytorch forecasting implementation. The current model was only trained on S&P500 data from current day constutuants and is therefore prone to survivorship bias. The following table displays the results of one evaluation run (seed=100) but was reproduced across 5 different seeds. The difference between ridge and ridge cfg is that standard ridge uses all available features while cfg uses the exact same feature set as the TFT model. The weights which gave these results is what the default checkpoint contains.

Metric                          |  Ridge h1 |Ridge-cfg h1   |PTF-TFT h1   |  Ridge h5 |Ridge-cfg h5  | PTF-TFT h5   | Ridge h20|Ridge-cfg h20 | PTF-TFT h20
--------------------------------|-----------|---------------|-------------|-----------|--------------|--------------|----------|--------------|-----------
IC (mean daily rank corr)       |  -0.0006  |    -0.0059    |   0.0107    |   0.0079  |     0.0087   |    0.0047    |   0.0059 |      0.0162  |     0.0054
IC t-stat (Newey-West)          |    -0.19  |      -1.41    |     2.30    |     2.05  |       1.53   |      0.50    |     0.93 |        2.24  |       0.30
R² vs zero forecast             | -1.3878%  |   -1.2294%    |-33.7192%    | -1.2946%  |   -0.9453%   |-100.3022%    | -1.4102% |    -1.1143%  | -492.9711%
MSE (raw units)                 |3.959e-04  |  3.953e-04    |5.222e-04    |1.919e-03  |  1.912e-03   | 3.795e-03    |7.588e-03 |   7.566e-03  |  4.437e-02
Top-bottom 20% spread           | -0.0052%  |   -0.0246%    |  0.0200%    |  0.1130%  |    0.1380%   |  -0.0136%    |  0.1441% |     0.3235%  |   -0.1385%
Long-short Sharpe (ann.)        |    -0.15  |      -0.46    |     0.34    |     0.62  |       0.34   |     -0.29    |     0.46 |        0.75  |      -0.18

Given these results and the low signal to noise ratio of large cap stocks, I
don't recomend trusting the model forecasts. 

## Data and licensing

This repository contains code and a trained model checkpoint only. It does not include or redistribute any market data. All data is downloaded by the user, on their own machine, directly from the providers below, and remains subject to each provider's own terms.

| Source | Used for | Access | Terms |
|---|---|---|---|
| Yahoo Finance, via [yfinance](https://github.com/ranaroussi/yfinance) | Prices, volume, market indices, earnings dates, news in the chat tools | No key | [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) |
| [FRED](https://fred.stlouisfed.org/) (St. Louis Fed) | Macro features such as credit spreads, yield curve slope, VIX and oil | Free API key | [FRED API terms](https://fred.stlouisfed.org/docs/api/terms_of_use.html) |
| [SEC EDGAR](https://www.sec.gov/edgar) | Company fundamentals from XBRL filings | No key, a declared user agent is required | [SEC access policy](https://www.sec.gov/os/webmaster-faq#developers) |

Things to be aware of:

- **Yahoo Finance data is for personal use.** yfinance is not affiliated with Yahoo, and Yahoo does not grant a license to redistribute or commercially use its data. Do not republish data you fetch, and check Yahoo's terms before using this project for anything beyond personal research.
- **Some FRED series carry third-party copyright.** Each series page states its source and any restrictions. Check them before redistributing any FRED data.
- **SEC EDGAR data is public.** Requests must send a descriptive user agent (`SEC_USER_AGENT` in `.env`) and stay under SEC's rate limit.
- The code is released under its own license (see `LICENSE`). That license does not cover any data, and no rights to third-party data are granted by this repository.
- The default checkpoint was trained on data from the sources above. It contains model weights only, not the underlying data.

This project is for research and education. It is not investment advice.

## Issues

- Depending on the configuration of your firewall it may block the container from communicating with any model running localy on the machine. Since compose creates it's own bridge network with a subnet, you may need to allow this subnet through the firewall. 

**1. Find the subnet while the stack is up**
```bash
docker network ls
```

```bash
docker network inspector simple-alpha_default | grep Subnet
```

**2. Add an exception for this network in your firewall**
