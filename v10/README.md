# Time Series AI for Real-Time Trading Risk Management

**Student:** Mohammad Naqi Zaidi | **ID:** B00963745  
**Programme:** B.Eng. Artificial Intelligence — Ulster University  
**Supervisors:** Dr Sheily Panwar, Dr Obada Al Dimashki

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn pyyaml

# 2. Prepare data directories
mkdir -p data/{market/eurusd/m1,market/eurusd/m5,context,events,detectors,execution,windows}
mkdir -p MQL5/Files/{commands,acks}
mkdir -p logs results

# 3. Place your data files:
#    - data/market/eurusd/m1/YYYYMMDD.csv   (MT5 M1 bars)
#    - data/events/forexfactory_calendar.csv (Forex Factory export)
#    - data/context/dxy.csv                  (Yahoo Finance DXY)
#    - data/windows/windows.json             (window manifest)

# 4. Run morning calibration
python -m v10.scripts.replay_window --config v10/config.yaml --window 2024-CPI-May

# 5. Generate results tables
python -m v10.scripts.make_results --config v10/config.yaml --results results/
```

## Repository Layout

```
v10/
├── config.yaml                 # Frozen parameters (Appendix K)
├── README.md
├── core/
│   ├── timeutils.py            # UTC helpers, ISO-8601, gap checks
│   ├── rolling.py              # Welford stats, ATR, EMA, CUSUM, MACD
│   ├── ingest.py               # MarketDataLoader, EventCalendarLoader, ContextLoader
│   └── features.py             # FeatureEngineer (incremental, online)
├── detectors/
│   ├── regime.py               # RegimeDetector (k-means + CUSUM gate + persistence)
│   ├── anomaly.py              # AnomalyDetector (z-scores + Isolation Forest)
│   └── calibrate.py            # Morning fit orchestration (06:00 UTC)
├── policy/
│   ├── state.py                # FSM: Allow/Throttle/Pause/HedgeActive + cooldowns
│   ├── rules.py                # DecisionPolicy (rule precedence + reason strings)
│   └── config.py               # YAML loader + config hash for audit
├── execution/
│   ├── mt5_io.py               # MT5Bridge (CSV command/ack channel)
│   └── audit.py                # AuditLogger + Evaluator (metrics computation)
└── scripts/
    ├── replay_window.py        # End-to-end window replay
    └── make_results.py         # Generate Tables 9-A through 9-D
```

## Module Map (matches UML class diagram)

| UML Class          | Module                     | Key Methods                              |
|--------------------|----------------------------|------------------------------------------|
| MarketDataLoader   | `core/ingest.py`           | fetch_latest, fetch_history, validate     |
| EventCalendarLoader| `core/ingest.py`           | load_events, make_event_flags             |
| ContextLoader      | `core/ingest.py`           | load_dxy, align_to_index                  |
| FeatureEngineer    | `core/features.py`         | update_features, build_model_input        |
| RegimeDetector     | `detectors/regime.py`      | fit, predict, confirm_transition          |
| AnomalyDetector    | `detectors/anomaly.py`     | score_z, fit_iforest, update              |
| DecisionPolicy     | `policy/rules.py`          | decide, make_reason                       |
| MT5Bridge          | `execution/mt5_io.py`      | write_command, read_ack, mirror_files     |
| AuditLogger        | `execution/audit.py`       | log_row, log_error                        |
| Evaluator          | `execution/audit.py`       | run_event_study, compute_metrics          |

## Configuration

All parameters are frozen in `config.yaml`. The three "single knob" packs are:

1. **Persistence pack:** `persistence.label`, `persistence.moderate`, `persistence.severe`
2. **Sensitivity pack:** `cusum.k_mult`, `cusum.h_mult`, `z_thresholds.moderate`, `z_thresholds.severe`
3. **Restraint pack:** `throttle.size_mult`, `throttle.duration_bars`, `pause.duration_bars`

Any change increments `config_version` and invalidates prior comparisons.

## Data Schemas

See Appendix A in the report. All timestamps are UTC ISO-8601.

## Reproducibility

- All random seeds fixed (seed=42)
- Calibration artifacts persisted in `data/detectors/regime_calibration.json`
- Window manifest frozen in `data/windows/windows.json`
- Config hash recorded in every audit row
