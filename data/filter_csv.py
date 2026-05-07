import pandas as pd
import json
from pathlib import Path

def filter_merge_csv(log_path: Path, out_path: Path, phase_data_path: Path):
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    phase_data = pd.read_csv(phase_data_path)[["timestamp", "phase"]]
    phase_data["timestamp"] = pd.to_datetime(phase_data["timestamp"], utc=True).dt.tz_localize(None)

    df = df.sort_values("timestamp")
    phase_data = phase_data.sort_values("timestamp")

    df = pd.merge_asof(
        df,
        phase_data,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=5),
    )

    df["phase"] = df["phase"].fillna("load_model")

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")