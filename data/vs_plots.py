"""Compare two CodeCarbon `.log` runs and save JPG plots.

The script ingests two `.log` files produced by the training pipeline.
Each file contains a mix of:

* Phase markers of the form
  ``phase=<name> energy=Measurement(time=<cumulative_seconds>, ...)`` emitted at
  the END of each phase (so the cumulative ``time`` is the phase boundary).
* Per-step JSON records logged by CodeCarbon (one per line) which carry
  per-component power, energy and utilization measurements.

For each (component, metric) combination the script renders one JPG with two
vertically stacked panels: the top panel shows the first log file and the
bottom panel shows the second log file (no overlay). Each panel keeps phase
shading and styling that mirrors ``data/plot_logs.py``.

Default output is 9 images (3 components x 3 metrics). Use ``--components``
and/or ``--metrics`` to restrict the set.

Usage::

    python data/vs_plots.py path/to/run_a.log path/to/run_b.log \
        --output-dir data/comparisons \
        --label-a "Ministral 14B QLoRA" --label-b "Ministral 14B LoRA"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


PHASE_RE = re.compile(r"^phase=(\S+)\s+energy=Measurement\(time=([0-9.]+)")


COMPONENT_LABELS: dict[str, str] = {
    "cpu": "CPU",
    "gpu": "GPU",
    "ram": "RAM",
}

# Metric family -> (column suffix, axis label suffix, descriptive title).
METRIC_FAMILIES: dict[str, dict] = {
    "power": {
        "column_suffix": "_power",
        "unit": "W",
        "title": "Power Consumption",
    },
    "energy": {
        "column_suffix": "_energy",
        "unit": "kWh",
        "title": "Cumulative Energy Consumed",
    },
    "utilization": {
        "column_suffix": "_utilization_percent",
        "unit": "%",
        "title": "Hardware Utilization",
    },
}


def parse_log_file(path: str | Path) -> pd.DataFrame:
    """Parse a CodeCarbon ``.log`` file into a DataFrame with phase labels.

    The returned frame is sorted by ``duration`` (elapsed seconds since the
    start of the run) and includes a ``phase`` column derived from the
    interleaved ``phase=...`` markers. Auxiliary metadata is attached via
    ``DataFrame.attrs``: ``hardware_info``, ``phase_blocks`` and ``source``.
    """
    path = Path(path)
    records: list[dict] = []
    phase_markers: list[tuple[str, float]] = []

    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = PHASE_RE.match(line)
            if m:
                phase_markers.append((m.group(1), float(m.group(2))))
                continue
            if line.startswith("{"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        raise ValueError(f"No JSON records found in {path}")

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("duration").reset_index(drop=True)

    phase_blocks: list[tuple[float, float, str]] = []
    prev_end = 0.0
    for name, end_time in phase_markers:
        phase_blocks.append((prev_end, end_time, name))
        prev_end = end_time

    last_duration = float(df["duration"].iloc[-1])
    if not phase_blocks:
        phase_blocks.append((0.0, last_duration, "unknown"))
    elif last_duration > phase_blocks[-1][1]:
        # Records continued past the last phase marker; extend the final phase.
        start, _, name = phase_blocks[-1]
        phase_blocks[-1] = (start, last_duration, name)

    def label_for(t: float) -> str:
        for start, end, name in phase_blocks:
            if start <= t <= end:
                return name
        return phase_blocks[-1][2]

    df["phase"] = df["duration"].apply(label_for)

    hardware_info = {
        "cpu": str(df["cpu_model"].iloc[0]) if "cpu_model" in df.columns else None,
        "gpu": str(df["gpu_model"].iloc[0]) if "gpu_model" in df.columns else None,
        "ram": (
            f"{float(df['ram_total_size'].iloc[0]):.1f} GB"
            if "ram_total_size" in df.columns
            else None
        ),
    }

    df.attrs["hardware_info"] = hardware_info
    df.attrs["phase_blocks"] = phase_blocks
    df.attrs["source"] = (
        str(df["project_name"].iloc[0]) if "project_name" in df.columns else path.stem
    )
    return df


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _draw_run_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    line_color,
    phase_palette: dict[str, tuple],
    panel_title: str,
    y_label: str,
    x_max: float,
    y_max: float,
    y_pad_frac: float,
) -> None:
    """Render a single run's series with phase shading on ``ax``."""
    ax.plot(
        df["duration"], df[column],
        color=line_color, linewidth=1.6,
    )
    for start, end, phase in df.attrs["phase_blocks"]:
        ax.axvspan(start, end, color=phase_palette[phase], alpha=0.2)

    ax.set_title(panel_title, loc="left", fontsize=11)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max * (1 + y_pad_frac) if y_max > 0 else 1)
    ax.set_xlabel("")


def plot_pair(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    component: str,
    metric: str,
    label_a: str = "A",
    label_b: str = "B",
    figsize: tuple[float, float] = (15, 9),
    y_pad_frac: float = 0.1,
    cmap_name: str = "tab10",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Render a stacked comparison plot for one ``(component, metric)`` pair.

    The top panel shows run A and the bottom panel shows run B for the same
    column. Both panels share the x-axis range and the y-axis range so the
    two runs are directly comparable.
    """
    if component not in COMPONENT_LABELS:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Expected one of: {list(COMPONENT_LABELS)}"
        )
    if metric not in METRIC_FAMILIES:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Expected one of: {list(METRIC_FAMILIES)}"
        )

    fam = METRIC_FAMILIES[metric]
    column = f"{component}{fam['column_suffix']}"

    for label, df in [(label_a, df_a), (label_b, df_b)]:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not present in run '{label}'")

    cmap = plt.get_cmap(cmap_name)
    color_a = cmap(0)
    color_b = cmap(3)

    # Shared phase palette across both runs so identical phase names share a color.
    seen_phases: list[str] = []
    for df in (df_a, df_b):
        for _, _, name in df.attrs["phase_blocks"]:
            if name not in seen_phases:
                seen_phases.append(name)
    phase_palette = {p: cmap((i + 1) % 10) for i, p in enumerate(seen_phases)}

    x_max = max(
        float(df_a["duration"].max()), float(df_b["duration"].max())
    )
    y_max = max(float(df_a[column].max()), float(df_b[column].max()))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    component_label = COMPONENT_LABELS[component]
    y_label = f"{component_label} {fam['title'].split()[0]} ({fam['unit']})"

    hw_a = df_a.attrs["hardware_info"].get(component)
    hw_b = df_b.attrs["hardware_info"].get(component)
    title_a = f"{label_a} (time: {_fmt_duration(df_a['duration'].max())})"
    if hw_a:
        title_a += f"  -  {hw_a}"
    title_b = f"{label_b} (time: {_fmt_duration(df_b['duration'].max())})"
    if hw_b:
        title_b += f"  -  {hw_b}"

    _draw_run_panel(
        axes[0], df_a, column, color_a, phase_palette,
        panel_title=title_a, y_label=y_label,
        x_max=x_max, y_max=y_max, y_pad_frac=y_pad_frac,
    )
    _draw_run_panel(
        axes[1], df_b, column, color_b, phase_palette,
        panel_title=title_b, y_label=y_label,
        x_max=x_max, y_max=y_max, y_pad_frac=y_pad_frac,
    )

    axes[-1].set_xlabel("Elapsed Time (s)")

    phase_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in phase_palette.values()
    ]
    phase_labels = list(phase_palette.keys())
    fig.legend(
        phase_handles,
        phase_labels,
        title="Phase",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    fig.suptitle(
        f"{component_label} {fam['title']}: {label_a} vs {label_b}"
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, list(axes)


def compare_logs(
    log_a: str | Path,
    log_b: str | Path,
    output_dir: str | Path = ".",
    label_a: str | None = None,
    label_b: str | None = None,
    components: Iterable[str] = tuple(COMPONENT_LABELS),
    metrics: Iterable[str] = tuple(METRIC_FAMILIES),
    show: bool = False,
) -> dict[tuple[str, str], Path]:
    """High-level entrypoint: parse both logs and write the requested JPGs."""
    df_a = parse_log_file(log_a)
    df_b = parse_log_file(log_b)

    label_a = label_a or df_a.attrs["source"]
    label_b = label_b or df_b.attrs["source"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[tuple[str, str], Path] = {}
    for component in components:
        for metric in metrics:
            save_path = output_dir / f"vs_{component}_{metric}.jpg"
            plot_pair(
                df_a, df_b, component, metric,
                label_a=label_a, label_b=label_b,
                save_path=save_path, show=show,
            )
            written[(component, metric)] = save_path
    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two CodeCarbon .log runs and save stacked JPG plots.",
    )
    parser.add_argument("log_a", type=Path, help="First .log file (top panel).")
    parser.add_argument("log_b", type=Path, help="Second .log file (bottom panel).")
    parser.add_argument("--label-a", default=None, help="Display label for run A.")
    parser.add_argument("--label-b", default=None, help="Display label for run B.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory in which the JPG plots are saved (default: current dir).",
    )
    parser.add_argument(
        "--components",
        nargs="*",
        choices=list(COMPONENT_LABELS.keys()),
        default=list(COMPONENT_LABELS.keys()),
        help="Components to plot (default: cpu gpu ram).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=list(METRIC_FAMILIES.keys()),
        default=list(METRIC_FAMILIES.keys()),
        help="Metric families to plot (default: power energy utilization).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each plot interactively in addition to saving it.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    written = compare_logs(
        args.log_a,
        args.log_b,
        output_dir=args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        components=args.components,
        metrics=args.metrics,
        show=args.show,
    )
    for (component, metric), path in written.items():
        print(f"[{component}/{metric}] saved -> {path}")


if __name__ == "__main__":
    main()
