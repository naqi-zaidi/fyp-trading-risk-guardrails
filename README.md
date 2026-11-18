# Trading Risk Guardrails

A real-time risk controller that sits between a trading strategy and MetaTrader 5. On every EUR/USD M1 bar it detects the volatility regime and any anomalies, then decides whether the strategy may trade at full size, at reduced size, or not at all. The decision goes to MT5 through a CSV command channel.

The controller never places trades itself. An MQL5 Expert Advisor reads each command, applies it, and writes back an acknowledgement. Every decision is logged with a human-readable reason and the hash of the config that produced it.

## Results

The controller was replayed bar by bar over two April 2026 macro-event windows.

| Window | Bars | Allow | Throttle | Pause | Hedge |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPI release | 152 | 94.7% | 5.3% | 0% | 0% |
| NFP release | 62 | 100% | 0% | 0% | 0% |

**Restraint on calm bars** (bars with no scheduled event and a calm or trending regime):

| Window | Calm bars | Throttle rate | Restraint threshold met |
| --- | ---: | ---: | :---: |
| CPI release | 135 | 5.9% | No |
| NFP release | 62 | 0.0% | Yes |

Example decision from the audit log:

```text
2026-05-13T04:56:18Z  EURUSD  throttle  size_mult=0.33
reason: regime trend_down, z_range=2.75, anomaly=moderate
```

## How it works

```text
MT5 M1 bars + event calendar + DXY context
        │
        ▼
Features: Welford rolling stats, ATR, EMA, MACD, z-scores (incremental, no lookahead)
        │
        ├──► Regime detector: k-means + CUSUM gate + persistence filter
        └──► Anomaly detector: rolling z-scores + Isolation Forest
        │
        ▼
Policy state machine: Allow → Throttle → Pause → Hedge, with cooldowns
        │
        ▼
CSV command ──► GuardrailBridgeEA.mq5 applies it ──► CSV acknowledgement ──► audit log
```

- **Regime detection** clusters volatility features with k-means. A CUSUM change-point gate and a persistence rule stop it from flipping on single-bar noise.
- **Anomaly scoring** combines rolling z-scores with an Isolation Forest that is refit each morning at 06:00 UTC.
- **The policy** is a finite-state machine with explicit rule precedence and cooldowns, so actions don't flicker bar to bar.
- **The MT5 bridge** uses plain CSV files. `OverlayPlotterEA.mq5` draws each regime and action on the chart.

## Run it

From the repository root:

```bash
pip install numpy pandas scikit-learn pyyaml pytest

python -m v10.scripts.morning_fit --config v10/config.yaml
python -m v10.scripts.replay_window --config v10/config.yaml --window 2026-CPI-Apr --output v10/results
python -m v10.scripts.make_results --config v10/config.yaml --results v10/results --output v10/results/tables
python -m pytest v10/tests
```

Sample EUR/USD M1 data for March to May 2026 is included in `v10/data/market/eurusd/m1/`. See `v10/SETUP_GUIDE.md` to attach the EAs in MT5.

## Engineering

- **Reproducible:** fixed seed (42), parameters frozen in `config.yaml`, and the config hash stamped on every audit row. Changing any parameter bumps the version.
- **Tested:** unit tests for rolling statistics, features, detectors, policy transitions (including the pause and hedge paths) and execution, plus an integration test.
- **Online by design:** features update incrementally per bar with Welford's algorithm, so nothing is recomputed over history and each decision uses only data up to the current bar.

## Limitations

- **Small sample.** Two event windows (214 bars) show the controller behaves sensibly. They are not statistical evidence that it reduces risk.
- **Over-throttling in calm markets.** The CPI window throttled 5.9% of calm bars and missed the restraint target, so the sensitivity parameters need tuning.
- **Untested escalation in replay.** Pause and hedge never fired in either window, so those paths are covered only by unit tests.
- **Latency.** The CSV bridge is polled, which adds seconds of delay. That's fine for M1 bars and unsuitable for anything faster.

## Repository layout

```text
v10/
├── core/        timeutils, rolling statistics, data ingest, features
├── detectors/   regime, anomaly, morning calibration
├── policy/      state machine, decision rules, config loader
├── execution/   MT5 bridge, audit logger, evaluator
├── mql5/        GuardrailBridgeEA.mq5, OverlayPlotterEA.mq5
├── scripts/     replay_window, make_results, morning_fit, backfill_bars
├── tests/       pytest suite
└── config.yaml  frozen parameters
```

---

Final Year Project, B.Eng. Artificial Intelligence, Ulster University. Mohammad Naqi Zaidi, supervised by Dr Sheily Panwar and Dr Obada Al Dimashki.
