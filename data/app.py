from filter_csv import filter_merge_csv
from plot_logs import plot_power_logs
from report import build_and_save_report
from pathlib import Path
import shutil
ASSETS_ROOT = Path("../../paper/assets")



def model_paths(slug: str, log_stem: str) -> dict[str, Path]:
    """Return all input/output paths for a given model run."""
    data = Path(slug)
    assets = ASSETS_ROOT / slug
    return {
        "log": data / f"{log_stem}.log",
        "csv": data / f"{log_stem}.csv",
        "loss_function_src": data / "loss_function.png",
        "phases": data / "power_timeseries.csv",
        "report_txt": data / "energy_report.txt",
        "report_json": data / "energy_report.json",
        "report_tex": data / "energy_report.tex",
        "power_pgf": assets / "power_consumption.pgf",
        "energy_pgf": assets / "energy_consumption.pgf",
        "util_pgf": assets / "utilization.pgf",
        "loss_function_dest": assets / "loss_function.png",
    }


if __name__ == "__main__":
    MODEL_NAME = "Mistral 7B QLoRA"
    MODEL_SLUG = "mistral/mistral-7b-qlora"
    LOG_STEM = "biomistral_7b_logs"
    TEX_PATH = "../../paper/reports/mistral/7b-qlora.tex"

    paths = model_paths(MODEL_SLUG, LOG_STEM)

    filter_merge_csv(paths["log"], paths["csv"], paths["phases"])

    build_and_save_report(
        csv_path=paths["csv"],
        txt_path=paths["report_txt"],
        json_path=paths["report_json"],
        latex_path=TEX_PATH,
    )

    plot_power_logs(
        csv_path=paths["csv"],
        title=f"Power Consumption over Time - {MODEL_NAME}",
        save_path=paths["power_pgf"],
    )
    plot_power_logs(
        csv_path=paths["csv"],
        title=f"Energy Consumption over Time - {MODEL_NAME}",
        metrics=[("cpu_energy", "CPU Energy (kWh)"), ("gpu_energy", "GPU Energy (kWh)"), ("ram_energy", "RAM Energy (kWh)")],
        y_padding=0,
        save_path=paths["energy_pgf"],
    )

    plot_power_logs(
        csv_path=paths["csv"],
        title=f"Utilization over Time - {MODEL_NAME}",
        metrics=[("cpu_utilization_percent", "CPU Utilization (%)"), ("gpu_utilization_percent", "GPU Utilization (%)"), ("ram_utilization_percent", "RAM Utilization (%)")],
        y_lims=[(0, 100), (0, 100), (0, 100)],
        save_path=paths["util_pgf"],
    )

    shutil.copy(paths["loss_function_src"], paths["loss_function_dest"])