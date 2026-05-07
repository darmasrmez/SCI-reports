"""Energy / power consumption report generation from CodeCarbon-style logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DEVICES = ("cpu", "gpu", "ram")

POWER_COLS = {d: f"{d}_power" for d in DEVICES}
ENERGY_COLS = {d: f"{d}_energy" for d in DEVICES}
UTIL_COLS = {d: f"{d}_utilization_percent" for d in DEVICES}


def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_first(df: pd.DataFrame, col: str) -> Any:
    if col in df.columns and not df[col].isna().all():
        return df[col].dropna().iloc[0]
    return None


def _device_stats(df: pd.DataFrame, device: str) -> dict[str, Any]:
    """Per-device aggregate stats over the full run."""
    power_col = POWER_COLS[device]
    energy_col = ENERGY_COLS[device]
    util_col = UTIL_COLS[device]

    stats: dict[str, Any] = {}

    if power_col in df.columns:
        p = df[power_col].dropna()
        stats["power_w"] = {
            "mean": float(p.mean()),
            "max": float(p.max()),
            "min": float(p.min()),
            "std": float(p.std()),
        }

    if energy_col in df.columns:
        e = df[energy_col].dropna()
        stats["energy_kwh"] = float(e.iloc[-1]) if len(e) else 0.0

    if util_col in df.columns:
        u = df[util_col].dropna()
        stats["utilization_percent"] = {
            "mean": float(u.mean()),
            "max": float(u.max()),
            "min": float(u.min()),
        }

    return stats


def _phase_breakdown(df: pd.DataFrame, phase_col: str = "phase") -> list[dict[str, Any]]:
    """Aggregate per phase, handling non-contiguous phase blocks correctly.

    Energy is attributed per-row using diffs of the cumulative columns and then
    summed by phase, so interleaved phases (e.g. fine_tuning / idle) don't get
    inflated by gaps. Duration is the sum of contiguous block durations.
    """
    if phase_col not in df.columns:
        return []

    df = df.sort_index()

    block_id = (df[phase_col] != df[phase_col].shift()).cumsum()

    block_durations = (
        df.assign(_b=block_id)
        .groupby("_b")
        .apply(
            lambda g: pd.Series(
                {
                    "phase": g[phase_col].iloc[0],
                    "duration_s": (g.index[-1] - g.index[0]).total_seconds(),
                    "samples": len(g),
                }
            )
        )
    )
    duration_per_phase = block_durations.groupby("phase")["duration_s"].sum()
    samples_per_phase = block_durations.groupby("phase")["samples"].sum()

    energy_per_phase: dict[str, dict[str, float]] = {
        str(p): {} for p in df[phase_col].dropna().unique()
    }
    for d in DEVICES:
        ec = ENERGY_COLS[d]
        if ec not in df.columns:
            continue
        deltas = df[ec].diff().fillna(0).clip(lower=0)
        summed = deltas.groupby(df[phase_col]).sum()
        for phase, value in summed.items():
            energy_per_phase.setdefault(str(phase), {})[d] = float(value)

    rows: list[dict[str, Any]] = []
    for phase in df[phase_col].dropna().unique():
        phase_str = str(phase)
        g = df[df[phase_col] == phase]
        duration_s = float(duration_per_phase.get(phase, 0.0))
        entry: dict[str, Any] = {
            "phase": phase_str,
            "samples": int(samples_per_phase.get(phase, len(g))),
            "duration_seconds": duration_s,
            "duration_hms": _fmt_hms(duration_s),
            "energy_kwh": energy_per_phase.get(phase_str, {}),
            "mean_power_w": {},
        }
        for d in DEVICES:
            pc = POWER_COLS[d]
            if pc in g.columns and len(g[pc].dropna()):
                entry["mean_power_w"][d] = float(g[pc].dropna().mean())
        rows.append(entry)
    return rows


def generate_report(
    csv_path: str | Path,
    timestamp_col: str = "timestamp",
    phase_col: str = "phase",
) -> dict[str, Any]:
    """Build a structured report dict from a merged log CSV."""
    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()

    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    total_seconds = (df.index[-1] - df.index[0]).total_seconds()
    total_hours = total_seconds / 3600.0

    energy_total_kwh = (
        float(df["energy_consumed"].iloc[-1])
        if "energy_consumed" in df.columns
        else sum(_device_stats(df, d).get("energy_kwh", 0.0) for d in DEVICES)
    )

    emissions_kg = (
        float(df["emissions"].iloc[-1])
        if "emissions" in df.columns
        else None
    )

    water_l = (
        float(df["water_consumed"].iloc[-1])
        if "water_consumed" in df.columns
        else None
    )

    avg_total_power_w = (
        (energy_total_kwh * 1000.0 * 3600.0) / total_seconds
        if total_seconds > 0
        else 0.0
    )

    devices = {d: _device_stats(df, d) for d in DEVICES}

    energy_share = {}
    if energy_total_kwh > 0:
        for d in DEVICES:
            e = devices[d].get("energy_kwh")
            if e is not None:
                energy_share[d] = 100.0 * e / energy_total_kwh

    report: dict[str, Any] = {
        "source_csv": str(Path(csv_path).resolve()),
        "run": {
            "project_name": _safe_first(df, "project_name"),
            "run_id": _safe_first(df, "run_id"),
            "experiment_id": _safe_first(df, "experiment_id"),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "samples": int(len(df)),
        },
        "duration": {
            "seconds": float(total_seconds),
            "hours": float(total_hours),
            "hms": _fmt_hms(total_seconds),
        },
        "totals": {
            "energy_consumed_kwh": energy_total_kwh,
            "emissions_kg_co2": emissions_kg,
            "water_consumed_l": water_l,
            "average_power_w": avg_total_power_w,
            "energy_share_percent": energy_share,
        },
        "hardware": {
            "cpu_model": _safe_first(df, "cpu_model"),
            "cpu_count": _safe_first(df, "cpu_count"),
            "gpu_model": _safe_first(df, "gpu_model"),
            "gpu_count": _safe_first(df, "gpu_count"),
            "ram_total_gb": (
                float(_safe_first(df, "ram_total_size"))
                if _safe_first(df, "ram_total_size") is not None
                else None
            ),
        },
        "environment": {
            "country": _safe_first(df, "country_name"),
            "region": _safe_first(df, "region"),
            "cloud_provider": _safe_first(df, "cloud_provider"),
            "cloud_region": _safe_first(df, "cloud_region"),
            "on_cloud": _safe_first(df, "on_cloud"),
            "pue": _safe_first(df, "pue"),
            "wue": _safe_first(df, "wue"),
            "os": _safe_first(df, "os"),
            "python_version": _safe_first(df, "python_version"),
            "codecarbon_version": _safe_first(df, "codecarbon_version"),
        },
        "devices": devices,
        "phases": _phase_breakdown(df, phase_col=phase_col),
    }

    return report


def format_report(report: dict[str, Any]) -> str:
    """Render the report dict as a human-readable text block."""
    lines: list[str] = []
    sep = "=" * 78

    run = report["run"]
    dur = report["duration"]
    tot = report["totals"]
    hw = report["hardware"]
    env = report["environment"]

    lines.append(sep)
    lines.append(f"ENERGY CONSUMPTION REPORT - {run.get('project_name') or 'unknown'}")
    lines.append(sep)
    lines.append(f"Run ID         : {run.get('run_id')}")
    lines.append(f"Experiment ID  : {run.get('experiment_id')}")
    lines.append(f"Window         : {run['start']}  ->  {run['end']}")
    lines.append(f"Samples        : {run['samples']}")
    lines.append(
        f"Duration       : {dur['hms']}  ({dur['hours']:.3f} h, {dur['seconds']:.1f} s)"
    )
    lines.append("")

    lines.append("-- Totals --")
    lines.append(f"  Energy consumed : {tot['energy_consumed_kwh']:.6f} kWh")
    if tot.get("emissions_kg_co2") is not None:
        lines.append(f"  CO2 emissions   : {tot['emissions_kg_co2']:.6f} kg")
    if tot.get("water_consumed_l") is not None:
        lines.append(f"  Water consumed  : {tot['water_consumed_l']:.6f} L")
    lines.append(f"  Avg total power : {tot['average_power_w']:.2f} W")
    lines.append("")

    lines.append("-- Hardware --")
    lines.append(f"  CPU  : {hw.get('cpu_model')}  (count={hw.get('cpu_count')})")
    lines.append(f"  GPU  : {hw.get('gpu_model')}  (count={hw.get('gpu_count')})")
    if hw.get("ram_total_gb") is not None:
        lines.append(f"  RAM  : {hw['ram_total_gb']:.2f} GB")
    lines.append("")

    lines.append("-- Environment --")
    loc = ", ".join(
        x for x in (env.get("region"), env.get("country")) if x
    ) or "n/a"
    cloud = (
        f"{env.get('cloud_provider')}/{env.get('cloud_region')}"
        if env.get("cloud_provider")
        else "n/a"
    )
    lines.append(f"  Location : {loc}")
    lines.append(f"  Cloud    : {cloud}  (on_cloud={env.get('on_cloud')})")
    lines.append(f"  PUE / WUE: {env.get('pue')} / {env.get('wue')}")
    lines.append("")

    lines.append("-- Per-device breakdown --")
    header = (
        f"  {'device':<6} {'energy(kWh)':>12} {'share(%)':>9} "
        f"{'mean P(W)':>10} {'max P(W)':>10} {'min P(W)':>10} "
        f"{'mean U(%)':>10} {'max U(%)':>9}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    share = tot.get("energy_share_percent", {})
    for d in DEVICES:
        ds = report["devices"].get(d, {})
        if not ds:
            continue
        e = ds.get("energy_kwh", 0.0)
        p = ds.get("power_w", {})
        u = ds.get("utilization_percent", {})
        lines.append(
            f"  {d.upper():<6} {e:>12.6f} {share.get(d, 0.0):>9.2f} "
            f"{p.get('mean', 0.0):>10.2f} {p.get('max', 0.0):>10.2f} {p.get('min', 0.0):>10.2f} "
            f"{u.get('mean', 0.0):>10.2f} {u.get('max', 0.0):>9.2f}"
        )
    lines.append("")

    if report.get("phases"):
        lines.append("-- Per-phase breakdown --")
        for ph in report["phases"]:
            lines.append(
                f"  [{ph['phase']}]  duration={ph['duration_hms']}  samples={ph['samples']}"
            )
            for d in DEVICES:
                e = ph["energy_kwh"].get(d)
                p = ph["mean_power_w"].get(d)
                if e is None and p is None:
                    continue
                e_s = f"{e:.6f} kWh" if e is not None else "  n/a"
                p_s = f"{p:.2f} W" if p is not None else " n/a"
                lines.append(f"      {d.upper():<3}  energy={e_s:>16}   mean power={p_s:>10}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex_escape(value: Any) -> str:
    if value is None:
        return "--"
    s = str(value)
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in s)


def _tex_num(value: Any, fmt: str = "{:.6f}", default: str = "--") -> str:
    """Format a number for siunitx's \\num{}; falls back to default if missing."""
    if value is None:
        return default
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return default


def format_report_latex(
    report: dict[str, Any],
    label_prefix: str = "tab:energy",
    standalone: bool = False,
) -> str:
    """Render the report as a LaTeX snippet.

    The output is meant to be ``\\input``-ed inside an existing document and
    expects the following packages to be loaded by the host preamble::

        \\usepackage{booktabs}
        \\usepackage{siunitx}
        \\usepackage{longtable}  % only if the per-phase table grows large

    If ``standalone=True`` a minimal compilable document wrapper is added.
    """
    run = report["run"]
    dur = report["duration"]
    tot = report["totals"]
    hw = report["hardware"]
    env = report["environment"]
    devices = report["devices"]
    phases = report.get("phases", [])
    share = tot.get("energy_share_percent", {})

    project = _tex_escape(run.get("project_name") or "unknown")

    parts: list[str] = []
    parts.append(f"% Auto-generated energy report for {project}")
    parts.append(f"% Source CSV: {_tex_escape(report.get('source_csv'))}")
    parts.append("% Required packages: booktabs, siunitx")
    parts.append("")

    parts.append(r"\begin{table}[htbp]")
    parts.append(r"  \centering")
    parts.append(
        rf"  \caption{{Resumen de experimento \texttt{{{project}}}.}}"
    )
    parts.append(rf"  \label{{{label_prefix}:summary-{project}}}")
    parts.append(r"  \begin{tabular}{ll}")
    parts.append(r"    \toprule")
    parts.append(r"    Campo & Valor \\")
    parts.append(r"    \midrule")
    parts.append(rf"    Inicio & {_tex_escape(run.get('start'))} \\")
    parts.append(rf"    Fin & {_tex_escape(run.get('end'))} \\")
    parts.append(
        rf"    Duración & {_tex_escape(dur['hms'])} "
        rf"(\SI{{{dur['hours']:.3f}}}{{\hour}}) \\"
    )
    parts.append(
        rf"    Energía consumida & \SI{{{_tex_num(tot.get('energy_consumed_kwh'), '{:.6f}')}}}"
        r"{\kilo\watt\hour} \\"
    )
    parts.append(
        rf"    Promedio de potencia & "
        rf"\SI{{{_tex_num(tot.get('average_power_w'), '{:.2f}')}}}{{\watt}} \\"
    )
    parts.append(rf"    CPU & {_tex_escape(hw.get('cpu_model'))} ($\times${hw.get('cpu_count')}) \\")
    parts.append(rf"    GPU & {_tex_escape(hw.get('gpu_model'))} ($\times${hw.get('gpu_count')}) \\")
    if hw.get("ram_total_gb") is not None:
        parts.append(
            rf"    RAM & \SI{{{hw['ram_total_gb']:.2f}}}{{\giga\byte}} \\"
        )
    loc_bits = [x for x in (env.get("region"), env.get("country")) if x]
    if loc_bits:
        parts.append(rf"    Ubicación & {_tex_escape(', '.join(loc_bits))} \\")


    parts.append(r"    \bottomrule")
    parts.append(r"  \end{tabular}")
    parts.append(r"\end{table}")
    parts.append("")

    parts.append(r"\begin{table}[htbp]")
    parts.append(r"  \centering")
    parts.append(r"  \footnotesize")
    parts.append(r"  \setlength{\tabcolsep}{4pt}")
    parts.append(
        rf"  \caption{{Estadísticas de energía y potencia por dispositivo para \texttt{{{project}}}.}}"
    )
    parts.append(rf"  \label{{{label_prefix}:devices-{project}}}")
    parts.append(
        r"  \begin{tabular}{l S[table-format=1.4] S[table-format=2.2] "
        r"S[table-format=3.2] S[table-format=3.2] S[table-format=3.2] "
        r"S[table-format=3.2] S[table-format=3.2]}"
    )
    parts.append(r"    \toprule")
    parts.append(
        r"    Dispositivo & {\makecell{Energía\\(\si{\kilo\watt\hour})}} & {\makecell{Participación\\(\%)}} & "
        r"{\makecell{Pot.\ media\\(\si{\watt})}} & {\makecell{Pot.\ máx.\\(\si{\watt})}} & {\makecell{Pot.\ min.\\(\si{\watt})}} & "
        r"{\makecell{Uso medio\\(\%)}} & {\makecell{Uso máx.\\(\%)}} \\"
    )
    parts.append(r"    \midrule")
    for d in DEVICES:
        ds = devices.get(d, {})
        if not ds:
            continue
        p = ds.get("power_w", {}) or {}
        u = ds.get("utilization_percent", {}) or {}
        parts.append(
            f"    {d.upper()} & "
            f"{_tex_num(ds.get('energy_kwh'), '{:.4f}', '{--}')} & "
            f"{_tex_num(share.get(d), '{:.2f}', '{--}')} & "
            f"{_tex_num(p.get('mean'), '{:.2f}', '{--}')} & "
            f"{_tex_num(p.get('max'), '{:.2f}', '{--}')} & "
            f"{_tex_num(p.get('min'), '{:.2f}', '{--}')} & "
            f"{_tex_num(u.get('mean'), '{:.2f}', '{--}')} & "
            f"{_tex_num(u.get('max'), '{:.2f}', '{--}')} \\\\"
        )
    parts.append(r"    \bottomrule")
    parts.append(r"  \end{tabular}")
    parts.append(r"\end{table}")
    parts.append("")

    if phases:
        parts.append(r"\begin{table}[htbp]")
        parts.append(r"  \centering")
        parts.append(r"  \footnotesize")
        parts.append(r"  \setlength{\tabcolsep}{4pt}")
        parts.append(
            rf"  \caption{{Estadísticas de energía y potencia por fase para \texttt{{{project}}}.}}"
        )
        parts.append(rf"  \label{{{label_prefix}:phases-{project}}}")
        parts.append(
            r"  \begin{tabular}{l l S[table-format=1.4] "
            r"S[table-format=1.4] S[table-format=1.4] "
            r"S[table-format=3.2] S[table-format=3.2] S[table-format=3.2]}"
        )
        parts.append(r"    \toprule")
        parts.append(
            r"    & & \multicolumn{3}{c}{Energía (\si{\kilo\watt\hour})} & "
            r"\multicolumn{3}{c}{Promedio de potencia (\si{\watt})} \\"
        )
        parts.append(r"    \cmidrule(lr){3-5} \cmidrule(lr){6-8}")
        parts.append(
            r"    Fase & Duración & {CPU} & {GPU} & {RAM} & "
            r"{Media} & {Máxima} & {Mínima} \\"
        )
        parts.append(r"    \midrule")
        for ph in phases:
            e = ph.get("energy_kwh", {}) or {}
            p = ph.get("mean_power_w", {}) or {}
            parts.append(
                f"    {_tex_escape(ph['phase'])} & "
                f"{_tex_escape(ph['duration_hms'])} & "
                f"{_tex_num(e.get('cpu'), '{:.4f}', '{--}')} & "
                f"{_tex_num(e.get('gpu'), '{:.4f}', '{--}')} & "
                f"{_tex_num(e.get('ram'), '{:.4f}', '{--}')} & "
                f"{_tex_num(p.get('cpu'), '{:.2f}', '{--}')} & "
                f"{_tex_num(p.get('gpu'), '{:.2f}', '{--}')} & "
                f"{_tex_num(p.get('ram'), '{:.2f}', '{--}')} \\\\"
            )
        parts.append(r"    \bottomrule")
        parts.append(r"  \end{tabular}")
        parts.append(r"\end{table}")
        parts.append("")

    body = "\n".join(parts)

    if standalone:
        body = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{siunitx}\n"
            "\\begin{document}\n"
            f"{body}\n"
            "\\end{document}\n"
        )

    return body


def save_report(
    report: dict[str, Any],
    txt_path: str | Path | None = None,
    json_path: str | Path | None = None,
    latex_path: str | Path | None = None,
    latex_standalone: bool = False,
) -> None:
    """Persist text, JSON, and/or LaTeX versions of the report."""
    if txt_path is not None:
        txt_path = Path(txt_path)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(format_report(report), encoding="utf-8")
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if latex_path is not None:
        latex_path = Path(latex_path)
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(
            format_report_latex(report, standalone=latex_standalone),
            encoding="utf-8",
        )


def build_and_save_report(
    csv_path: str | Path,
    txt_path: str | Path | None = None,
    json_path: str | Path | None = None,
    latex_path: str | Path | None = None,
    latex_standalone: bool = False,
    print_report: bool = True,
) -> dict[str, Any]:
    """Convenience: generate, optionally print, and save in one call."""
    report = generate_report(csv_path)
    if print_report:
        print(format_report(report))
    if txt_path or json_path or latex_path:
        save_report(
            report,
            txt_path=txt_path,
            json_path=json_path,
            latex_path=latex_path,
            latex_standalone=latex_standalone,
        )
    return report
