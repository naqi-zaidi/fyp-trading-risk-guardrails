# Setup Guide — Time Series AI for Real-Time Trading Risk Management

Complete step-by-step instructions to get the guardrail system running from scratch on a Windows machine with MT5.

---

## Prerequisites

Before you start, make sure you have:

- **Windows 10/11** (MT5 is Windows-only)
- **Python 3.10+** installed and on PATH ([python.org](https://www.python.org/downloads/))
- **MetaTrader 5** demo account set up with any broker (e.g., ICMarkets, Pepperstone, XM)
- A text editor (VS Code recommended)

---

## Phase 1 — Project Structure

### Step 1: Create the project folder

Open a terminal (PowerShell or CMD) and run:

```powershell
mkdir C:\Projects\v10
cd C:\Projects\v10
```

### Step 2: Place the downloaded files

Copy the entire `v10/` folder you downloaded from Claude into `C:\Projects\`. Your structure should look like:

```
C:\Projects\v10\
├── config.yaml
├── controller.py
├── README.md
├── core\
│   ├── __init__.py
│   ├── timeutils.py
│   ├── rolling.py
│   ├── ingest.py
│   └── features.py
├── detectors\
│   ├── __init__.py
│   ├── regime.py
│   ├── anomaly.py
│   └── calibrate.py
├── policy\
│   ├── __init__.py
│   ├── state.py
│   ├── rules.py
│   └── config.py
├── execution\
│   ├── __init__.py
│   ├── mt5_io.py
│   └── audit.py
├── charting\
│   └── mt5\
│       ├── __init__.py
│       └── overlay_writer.py
├── scripts\
│   ├── __init__.py
│   ├── replay_window.py
│   ├── make_results.py
│   ├── morning_fit.py
│   └── backfill_bars.py
└── mql5\
    ├── GuardrailBridgeEA.mq5
    └── OverlayPlotterEA.mq5
```

### Step 3: Create the data directories

```powershell
cd C:\Projects\v10
mkdir data\market\eurusd\m1
mkdir data\market\eurusd\m5
mkdir data\context
mkdir data\events
mkdir data\detectors
mkdir data\execution
mkdir data\windows
mkdir data\macro
mkdir logs
mkdir results
```

---

## Phase 2 — Python Environment

### Step 4: Create a virtual environment (recommended)

```powershell
cd C:\Projects
python -m venv venv
.\venv\Scripts\activate
```

You should see `(venv)` in your prompt. Every time you open a new terminal, run `.\venv\Scripts\activate` again.

### Step 5: Install dependencies

```powershell
pip install numpy pandas scikit-learn pyyaml MetaTrader5
```

**Note:** `MetaTrader5` only works on Windows. If you get an error, make sure you're using a 64-bit Python installation.

### Step 6: Verify the installation

```powershell
cd C:\Projects
python -c "from v10.core.rolling import WelfordRolling; print('OK: all imports work')"
```

If you see `OK: all imports work`, you're good.

---

## Phase 3 — MetaTrader 5 Setup

### Step 7: Enable Algo Trading in MT5

1. Open MetaTrader 5
2. Go to **Tools → Options → Expert Advisors**
3. Check: **Allow algorithmic trading**
4. Check: **Allow DLL imports** (needed for file I/O)
5. Click OK

### Step 8: Find your MT5 data folder

In MT5, go to **File → Open Data Folder**. This opens a folder like:

```
C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX\
```

Note this path — you'll need it. Inside it you'll see a `MQL5\` folder.

### Step 9: Create the bridge directories

Inside your MT5 data folder:

```powershell
# Replace the path below with YOUR MT5 data folder path
cd "C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX"
mkdir MQL5\Files\commands
mkdir MQL5\Files\acks
mkdir MQL5\Files\overlays
```

### Step 10: Install the Expert Advisors

Copy the `.mq5` files to the MT5 Experts folder:

```powershell
copy C:\Projects\v10\mql5\GuardrailBridgeEA.mq5 "C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX\MQL5\Experts\"
copy C:\Projects\v10\mql5\OverlayPlotterEA.mq5 "C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX\MQL5\Experts\"
```

### Step 11: Compile the EAs

1. In MT5, open **MetaEditor** (F4 or the icon in toolbar)
2. In the Navigator panel, find `Experts\GuardrailBridgeEA.mq5`
3. Double-click to open it
4. Press **F7** (Compile). You should see "0 error(s)" in the output
5. Repeat for `OverlayPlotterEA.mq5`
6. Close MetaEditor

### Step 12: Update config.yaml with your MT5 paths

Open `C:\Projects\v10\config.yaml` in a text editor. Find the `paths:` section and update the MT5 paths to match your terminal:

```yaml
paths:
  # ... keep the data paths as they are ...
  
  # Update these to YOUR MT5 data folder:
  mt5_commands: "C:/Users/YourName/AppData/Roaming/MetaQuotes/Terminal/XXXXXXXX/MQL5/Files/commands/control.csv"
  mt5_acks: "C:/Users/YourName/AppData/Roaming/MetaQuotes/Terminal/XXXXXXXX/MQL5/Files/acks/ack.csv"
```

Use forward slashes `/` in the paths (Python handles them on Windows).

---

## Phase 4 — Data Preparation

### Step 13: Export EURUSD M1 bars from MT5

**Option A — Python API (automatic):**

```powershell
cd C:\Projects
python -m v10.scripts.backfill_bars --symbol EURUSD --timeframe M1 --days 60 --output data/market/eurusd/m1
```

This connects to your running MT5 terminal and downloads 60 days of M1 bars.

**Option B — Manual export from History Center:**

1. In MT5, press **F2** to open the History Center
2. Select **EURUSD** → **1 Minute**
3. Make sure data is loaded (click "Request" if needed)
4. You'll need to export this data. Use MT5's built-in Python:

```python
# Run this in a Python script while MT5 is open
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone

mt5.initialize()
utc_now = datetime.now(timezone.utc)
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M1,
                              utc_now - timedelta(days=60), utc_now)
df = pd.DataFrame(rates)
df["ts_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
df = df.rename(columns={"tick_volume": "volume"})
df["symbol"] = "EURUSD"

# Write day-partitioned files
for day, group in df.groupby(df["ts_utc"].dt.strftime("%Y%m%d")):
    out = group[["ts_utc", "open", "high", "low", "close", "volume", "symbol"]].copy()
    out["ts_utc"] = out["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(f"data/market/eurusd/m1/{day}.csv", index=False)
    print(f"Wrote {day}.csv ({len(out)} bars)")

mt5.shutdown()
```

### Step 14: Prepare the Forex Factory event calendar

1. Go to [forexfactory.com/calendar](https://www.forexfactory.com/calendar)
2. Set the date range to cover your data (last 60+ days)
3. Export or manually create a CSV file with these columns:

```csv
ts_utc,currency,event_name,actual,forecast,previous,impact
2024-05-14T12:30:00Z,USD,CPI m/m,0.3%,0.4%,0.4%,High
2024-06-07T12:30:00Z,USD,Non-Farm Employment Change,272K,185K,165K,High
```

Save it as: `C:\Projects\v10\data\events\forexfactory_calendar.csv`

**Important:** The `impact` column should be `High`, `Medium`, or `Low` (the loader maps these to 3, 2, 1 automatically).

If you have any manual override events, create:
`C:\Projects\v10\data\events\manual_overrides.csv` with the same format.

### Step 15: Prepare DXY context data (optional but recommended)

Download DXY data from Yahoo Finance. You can do this with a quick script:

```python
# Run this once to get DXY context data
import yfinance as yf
# pip install yfinance if needed

dxy = yf.download("DX-Y.NYB", period="90d", interval="1d")
dxy = dxy.reset_index()
dxy.columns = ["ts_utc", "open", "high", "low", "close", "adj_close", "volume"]
dxy["symbol"] = "DXY"
dxy = dxy[["ts_utc", "open", "high", "low", "close", "volume", "symbol"]]
dxy["ts_utc"] = dxy["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
dxy.to_csv("data/context/dxy.csv", index=False)
print(f"Wrote {len(dxy)} DXY rows")
```

If you skip this step, the system works fine — it just won't have DXY context.

### Step 16: Create the window manifest

Create the file `data\windows\windows.json` that defines your evaluation windows. Here's a template with common events — adjust the dates to match your actual data range:

```json
{
  "version": "1.0",
  "windows": [
    {
      "code": "2024-CPI-May",
      "start_utc": "2024-05-14T11:30:00Z",
      "end_utc": "2024-05-14T15:00:00Z",
      "notes": "US CPI release and immediate aftermath",
      "family": "macro_event"
    },
    {
      "code": "2024-NFP-Sep",
      "start_utc": "2024-09-06T11:30:00Z",
      "end_utc": "2024-09-06T15:00:00Z",
      "notes": "US Nonfarm Payrolls",
      "family": "macro_event"
    },
    {
      "code": "2024-ECB-Oct",
      "start_utc": "2024-10-17T11:45:00Z",
      "end_utc": "2024-10-17T15:00:00Z",
      "notes": "ECB rate decision and press conference",
      "family": "macro_event"
    },
    {
      "code": "2024-Quiet-Jun",
      "start_utc": "2024-06-10T06:00:00Z",
      "end_utc": "2024-06-14T20:59:00Z",
      "notes": "Summer quiet week, no high-impact events",
      "family": "calm_week"
    }
  ]
}
```

---

## Phase 5 — First Run (Calibration + Replay)

### Step 17: Verify your data is readable

```powershell
cd C:\Projects
python -c "
from v10.core.ingest import MarketDataLoader
m = MarketDataLoader('data/market/eurusd', timeframe='m1')
df = m.fetch_latest()
print(f'Latest file: {len(df)} bars')
print(df.head(3))
"
```

You should see your bar data printed. If you get a FileNotFoundError, check that your CSV files are in `data\market\eurusd\m1\` with filenames like `20240514.csv`.

### Step 18: Run morning calibration

This fits the k-means regime detector on your historical data:

```powershell
cd C:\Projects
python -m v10.scripts.morning_fit --config v10/config.yaml
```

You should see output like:

```
Starting morning calibration for 20240601
Regime calibration complete: 5 clusters on XXXXX bars
Isolation Forest fitted on XXXXX samples
Morning calibration complete (version=20240601)
```

This creates `data\detectors\regime_calibration.json` — the frozen centroids for today.

### Step 19: Run a window replay

Pick one of the windows from your manifest that falls within your data range:

```powershell
python -m v10.scripts.replay_window --config v10/config.yaml --window 2024-CPI-May
```

This runs the full pipeline — features, detectors, policy — bar-by-bar over the window and writes:
- `results\2024-CPI-May_decisions.csv` — every bar's features, labels, and action
- `logs\YYYYMMDD\decisions.csv` — audit trail

### Step 20: Generate results tables

```powershell
python -m v10.scripts.make_results --config v10/config.yaml --results results/
```

This reads all `*_decisions.csv` files in `results/` and generates:
- `results\tables\table_9C_restraint.csv` — calm-market pause/throttle rates
- `results\tables\action_summary.csv` — action distribution per window
- `results\tables\aggregate_metrics.csv` — overall summary

---

## Phase 6 — MT5 Live Integration

### Step 21: Attach the GuardrailBridgeEA

1. In MT5, open a **EURUSD M1** chart
2. In the Navigator panel (Ctrl+N), expand **Expert Advisors**
3. Drag **GuardrailBridgeEA** onto the chart
4. In the settings dialog:
   - **Common tab:** Check "Allow Algo Trading"
   - **Inputs tab:** Verify the symbol is `EURUSD` and lot size is correct for your demo
5. Click OK

You should see "GuardrailBridgeEA initialized" in the Experts tab (Ctrl+T → Experts).

### Step 22: Attach the OverlayPlotterEA (optional)

On the same or a second EURUSD M1 chart:
1. Drag **OverlayPlotterEA** onto the chart
2. Accept the defaults
3. Click OK

This EA polls `MQL5\Files\overlays\` and draws S/R lines, regime state panels, etc.

### Step 23: Run the Python controller

Open a new terminal, activate your venv, and start the controller:

```powershell
cd C:\Projects
.\venv\Scripts\activate
python -m v10.controller --config v10/config.yaml
```

The controller is now ready. In a full live setup, you would modify `controller.py` to poll MT5 for new bars every minute using the MT5 Python API. The key integration point is the `process_bar()` function:

```python
import MetaTrader5 as mt5
import time

mt5.initialize()
while True:
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 1)
    if rates is not None and len(rates) > 0:
        bar = {
            "open": float(rates[0]["open"]),
            "high": float(rates[0]["high"]),
            "low": float(rates[0]["low"]),
            "close": float(rates[0]["close"]),
            "volume": float(rates[0]["tick_volume"]),
            "ts_utc": pd.Timestamp(rates[0]["time"], unit="s", tz="UTC").isoformat(),
        }
        decision = process_bar(bar, pipeline, cfg)
        print(f"{bar['ts_utc']} | {decision['action']} | {decision['reason_code']}")
    time.sleep(60)
```

---

## Phase 7 — Running the Full Evaluation

### Step 24: Replay all windows

Run each window from your manifest:

```powershell
python -m v10.scripts.replay_window --config v10/config.yaml --window 2024-CPI-May
python -m v10.scripts.replay_window --config v10/config.yaml --window 2024-NFP-Sep
python -m v10.scripts.replay_window --config v10/config.yaml --window 2024-ECB-Oct
python -m v10.scripts.replay_window --config v10/config.yaml --window 2024-Quiet-Jun
```

Or automate it:

```powershell
$windows = @("2024-CPI-May", "2024-NFP-Sep", "2024-ECB-Oct", "2024-Quiet-Jun")
foreach ($w in $windows) {
    python -m v10.scripts.replay_window --config v10/config.yaml --window $w
}
```

### Step 25: Generate final results

```powershell
python -m v10.scripts.make_results --config v10/config.yaml --results results/ --output results/tables
```

### Step 26: Verify your pass criteria

Open `results\tables\table_9C_restraint.csv` and check:
- **Pause rate < 1%** in calm windows
- **Throttle rate < 5%** in calm windows

Open `results\tables\action_summary.csv` for the overall action distribution.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'v10'"

Make sure you're running from `C:\Projects` (the parent of v10), not from inside v10:

```powershell
cd C:\Projects
python -m v10.scripts.replay_window ...
```

### "FileNotFoundError: No CSV files in data\market\eurusd\m1"

Your bar CSVs need to be named `YYYYMMDD.csv` (e.g., `20240514.csv`) and have the header:

```
ts_utc,open,high,low,close,volume,symbol
```

### MT5 EA not picking up commands

1. Check that `MQL5\Files\commands\control.csv` exists and has data
2. Check the Experts tab in MT5 for error messages
3. Make sure Algo Trading is enabled (the button in the toolbar should be green)
4. Verify the file paths in `config.yaml` match your MT5 data folder

### "sklearn not available" warning

Install scikit-learn: `pip install scikit-learn`. Without it, the Isolation Forest is disabled (z-score anomaly detection still works).

### Calibration produces unreliable results

You need at least 30 calendar days (around 30,000+ M1 bars) for good calibration. If you have less data, the system warns you but still runs. Get more data using `backfill_bars.py`.

---

## File Checklist

Before submission, verify you have these deliverables:

```
v10/
├── config.yaml                          ✓ Frozen parameters
├── controller.py                        ✓ Main entry point
├── README.md                            ✓ Documentation
├── core/timeutils.py                    ✓ UTC helpers
├── core/rolling.py                      ✓ Welford, ATR, CUSUM, MACD
├── core/ingest.py                       ✓ Data loaders (3 classes)
├── core/features.py                     ✓ Feature engineering
├── detectors/regime.py                  ✓ k-means + CUSUM gate
├── detectors/anomaly.py                 ✓ z-score + Isolation Forest
├── detectors/calibrate.py               ✓ Morning fit orchestration
├── policy/state.py                      ✓ FSM + cooldowns
├── policy/rules.py                      ✓ Decision policy rules
├── policy/config.py                     ✓ YAML config loader
├── execution/mt5_io.py                  ✓ CSV bridge to MT5
├── execution/audit.py                   ✓ Audit logger + evaluator
├── charting/mt5/overlay_writer.py       ✓ Chart overlay exporter
├── scripts/replay_window.py             ✓ Window replay
├── scripts/make_results.py              ✓ Results tables
├── scripts/morning_fit.py               ✓ Daily calibration
├── scripts/backfill_bars.py             ✓ MT5 data import
├── mql5/GuardrailBridgeEA.mq5          ✓ MT5 bridge EA
├── mql5/OverlayPlotterEA.mq5           ✓ Chart overlay EA
├── data/detectors/regime_calibration.json  ✓ Calibration artifact
├── data/windows/windows.json            ✓ Frozen window manifest
├── results/tables/*.csv                 ✓ Evaluation results
└── logs/YYYYMMDD/decisions.csv          ✓ Audit trail
```

Total: **18 Python modules + 2 MQL5 EAs + config + README = ~5,000 lines**
