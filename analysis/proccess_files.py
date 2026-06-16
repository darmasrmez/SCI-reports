import pandas as pd
import numpy as np
from pathlib import Path
import os


def search_files(pattern:str, path:Path) -> list[Path]:
    return list(path.glob(f"**/*/*{pattern}*"))


METRIC_SPECS: list[tuple[str, str, str, tuple[str, ...]]] = [
    ('cpu_power',                'cpu', 'power',       ('mean', 'max', 'min', 'std')),
    ('gpu_power',                'gpu', 'power',       ('mean', 'max', 'min', 'std')),
    ('ram_power',                'ram', 'power',       ('mean', 'max', 'min', 'std')),
    ('cpu_utilization_percent',  'cpu', 'utilization', ('mean', 'max', 'min')),
    ('gpu_utilization_percent',  'gpu', 'utilization', ('mean', 'max', 'min')),
    ('ram_utilization_percent',  'ram', 'utilization', ('mean', 'max', 'min')),
    ('ram_used_gb',              'ram', 'used_gb',     ('mean', 'max', 'min')),
]


def process_df(df: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    row: dict[str, object] = {'model': model} if model is not None else {}
    for col, prefix, suffix, aggs in METRIC_SPECS:
        stats = df[col].agg(list(aggs))
        for agg in aggs:
            row[f'{prefix}_{agg}_{suffix}'] = stats[agg]
    return pd.DataFrame([row])


def process_logs(file:Path) -> pd.DataFrame:
    df = pd.read_csv(file)
    model = file.parent.parent.name
    size = file.parent.name
    df["model"] = model+"-"+size
    cols = ['project_name', 'model', 'duration', 'cpu_power', 'gpu_power', 'ram_power', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model', 'ram_total_size', 'cpu_utilization_percent', 'gpu_utilization_percent', 'ram_utilization_percent', 'ram_used_gb']
    return df[cols]


def search_emissions(path:Path) -> list[Path]:
    """Search for emissions files in the given path."""
    return list(path.glob(f"**/*/emissions.csv"))


def process_csv(file:Path):
    df = pd.read_csv(file)
    model = file.parent.parent.name
    size = file.parent.name
    df["model"] = model+"-"+size
    return df

if __name__ == "__main__":
    # emissions_files = search_emissions(Path("../data"))
    # df = pd.DataFrame()
    # for file in emissions_files:
    #     df_temp = process_csv(file)
    #     df = pd.concat([df, df_temp], ignore_index=True)
    # df.to_csv("experiments_emissions.csv", index=False)
    # print(df.head())
    # print(df.shape)

    logs_files = search_files("logs.csv", Path("../data"))
    for file in logs_files:
        df = process_logs(file)
        df_stats = process_df(df)

        print(df_stats)
        df_stats.to_csv(f"logs_stats_{file.parent.parent.name}.csv", index=False)
        break