import matplotlib.pyplot as plt
import pandas as pd

METRICS = [
    ("cpu_w", "CPU Power (W)"),
    ("ram_w", "RAM Power (W)"),
    ("gpu_e", "GPU Energy (kWh)"),
]

df = pd.read_csv('./power_timeseries.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()

phases = df['phase'].unique()
cmap = plt.get_cmap('tab10')
phase_colors = {phase: cmap(i) for i, phase in enumerate(phases)}

phase_changes = (df['phase'] != df['phase'].shift()).cumsum()
phase_blocks = [
    (group.index[0], group.index[-1], group['phase'].iloc[0])
    for _, group in df.groupby(phase_changes)
]

total_duration = df.index[-1] - df.index[0]
total_seconds = int(total_duration.total_seconds())
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

fig, axes = plt.subplots(len(METRICS), 1, figsize=(15, 9), sharex=True)

for ax, (col, ylabel) in zip(axes, METRICS):
    df[col].plot(ax=ax)
    for start, end, phase in phase_blocks:
        ax.axvspan(start, end, color=phase_colors[phase], alpha=0.2)

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_ylim(0, df[col].max() * 1.1 if df[col].max() > 0 else 1)
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

fig.suptitle("Power Consumption over Time")
plt.tight_layout()
plt.show()