import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METRICS = [
    ("cpu_power", "CPU Power (W)"),
    ("ram_power", "RAM Power (W)"),
    ("gpu_power", "GPU Power (W)"),
]


def plot_power_logs(
    csv_path,
    title="Power Consumption over Time",
    metrics=DEFAULT_METRICS,
    timestamp_col="timestamp",
    phase_col="phase",
    figsize=(15, 9),
    cmap_name="tab10",
    y_padding=10,
    y_lims=None,
    save_path=None,
    show=True,
):
    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()

    phases = df[phase_col].unique()
    cmap = plt.get_cmap(cmap_name)
    phase_colors = {phase: cmap(i) for i, phase in enumerate(phases)}

    phase_changes = (df[phase_col] != df[phase_col].shift()).cumsum()
    phase_blocks = [
        (group.index[0], group.index[-1], group[phase_col].iloc[0])
        for _, group in df.groupby(phase_changes)
    ]

    total_duration = df.index[-1] - df.index[0]
    total_seconds = int(total_duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    hardware_info = {
        "cpu": str(df["cpu_model"].iloc[0]) if "cpu_model" in df.columns else None,
        "gpu": str(df["gpu_model"].iloc[0]) if "gpu_model" in df.columns else None,
        "ram": (
            f"{float(df['ram_total_size'].iloc[0]):.1f} GB"
            if "ram_total_size" in df.columns
            else None
        ),
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for i, (ax, (col, ylabel)) in enumerate(zip(axes, metrics)):
        df[col].plot(ax=ax)
        for start, end, phase in phase_blocks:
            ax.axvspan(start, end, color=phase_colors[phase], alpha=0.2)

        hw_key = col.split("_")[0].lower()
        hw = hardware_info.get(hw_key)
        if hw:
            ylabel = f"{ylabel}\n{hw}"
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(df.index[0], df.index[-1])

        if y_lims is not None:
            ylim = y_lims[i] if isinstance(y_lims, (list, tuple)) and y_lims and isinstance(y_lims[0], (list, tuple)) else y_lims
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, df[col].max() + y_padding)
        ax.set_xlabel("")

    axes[-1].set_xlabel("Time")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.2) for c in phase_colors.values()]
    labels = list(phase_colors.keys())

    duration_handle = plt.Line2D([0], [0], color="none")
    handles.append(duration_handle)
    labels.append(f"Total time: {duration_str}")

    fig.legend(
        handles,
        labels,
        title="Phase",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes