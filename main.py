import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

FESEN = 444 # Region-specific carbon intensity for Mexico(gCO2eq/kWh).
VIRGINIA = 267.6195 # Region-specific carbon intensity for Virginia(gCO2eq/kWh).
IOWA = 635.94 # Region-specific carbon intensity for Iowa(gCO2eq/kWh).
ARIZONA = 336.0213 # Region-specific carbon intensity for Arizona(gCO2eq/kWh).

def operational_energy(electricity_consumption, region)->float:
    """
    Calculate the operational energy for a given electricity consumption and region factor.
    Args:
        electricity_consumption: energy consumed by a software system for a functional unit of work, kilowatt hours (kWh).
        region: The region of the system, Units: this shall be in grams of carbon per kilowatt hours (gCO2eq/kWh).
    Returns:
        The operational energy of the system.
    """
    region_factor = {
        "virginia": VIRGINIA,
        "Iowa": IOWA,
        "mexico city": FESEN,
        "arizona": ARIZONA,
    }
    return electricity_consumption * region_factor[region]


def embodied_emissions(total_embodied:float, time_reserved:float, resource_reserved:float, expected_lifespan:float, total_resources:float)->float:
    """
    Calculate the embodied emissions for a given total embodied, time reserved, resource reserved, expected lifespan, and total resources.
    Args:
        total_embodied: The total embodied emissions of the system, Units: this shall be in grams of carbon per kilowatt hours (gCO2eq/kWh).
        time_reserved: The time reserved for the system, Units: this shall be in hours.
        resource_reserved: The resource reserved for the system.
        expected_lifespan: The expected lifespan of the system, Units: this shall be in hours.
        total_resources: The total resource of the system.
    Returns:
        The embodied emissions of the system.
    """
    time_share = time_reserved / expected_lifespan
    resource_share = resource_reserved / total_resources
    return total_embodied * time_share * resource_share

def sci_score(operational, embodied)->float:
    return operational + embodied



def search_emissions(path:Path) -> list[Path]:
    """Search for emissions files in the given path."""
    return list(path.glob(f"**/*/emissions.csv"))


def latex_embodied_emissions(device, total_embodied, time_reserved, resource_reserved, expected_lifespan, total_resources, embodied_total)->str:
    return f"M_{ {device} } = {total_embodied} \\times \\frac{{ {time_reserved} }}{{ {expected_lifespan} }} \\times \\frac{{ {resource_reserved} }}{{ {total_resources} }} = {embodied_total}"



def make_latex_formula(energy_consumed, region, total_embodied, sci_ans)->str:
    region_factor = {
        "virginia": VIRGINIA,
        "Iowa": IOWA,
        "mexico city": FESEN,
        "arizona": ARIZONA,
    }
    return f"SCI = ({energy_consumed} \\times {region_factor[region]}) + {total_embodied} = {sci_ans}"




if __name__ == "__main__":


    emissions_files = search_emissions(Path("./data"))
    hardware_info = pd.read_csv(Path("./embodied.csv"))
    df_sci_scores = pd.DataFrame(columns=["project_name", "sci_score"])

    for file in emissions_files:
        logs_model = pd.read_csv(file)
        energy_consumed = logs_model["energy_consumed"].iloc[0]
        region = logs_model["region"].iloc[0]
        operational_emissions = operational_energy(energy_consumed, region)

        project_name = logs_model["project_name"].iloc[0]
        print("Project name: ", project_name)
        cpu_model = logs_model["cpu_model"].iloc[0]
        cpu_count = logs_model["cpu_count"].iloc[0]
        gpu_model = logs_model["gpu_model"].iloc[0]
        gpu_count = logs_model["gpu_count"].iloc[0]
        ram_gb = float(logs_model["ram_total_size"].iloc[0])
        ram_used_gb = float(logs_model["ram_used_gb"].iloc[0])


        gpu_model = gpu_model[4:]

        duration_s = logs_model["duration"].iloc[0]

        duration_h = duration_s / 3600

        # print("CPU model: ", cpu_model)
        # print("GPU model: ", gpu_model)
        # print("cpu count: ", cpu_count)
        # print("gpu count: ", gpu_count)

        cpu_embodied = (hardware_info["total_embodied"].loc[hardware_info["device"] == cpu_model].values[0])*1000
        gpu_embodied = (hardware_info["total_embodied"].loc[hardware_info["device"] == gpu_model].values[0])*1000*gpu_count
        ram_embodied = ram_gb * 2500
        cpu_lifespan = hardware_info["lifespan_years"].loc[hardware_info["device"] == cpu_model].values[0]*8760
        gpu_lifespan = hardware_info["lifespan_years"].loc[hardware_info["device"] == gpu_model].values[0]*8760
        ram_lifespan = 5*8760
        cpu_cores = hardware_info["cores_per_device"].loc[hardware_info["device"] == cpu_model].values[0]
        gpu_cores = hardware_info["cores_per_device"].loc[hardware_info["device"] == gpu_model].values[0]

        if cpu_count == 128:
            cpu_cores = 128


        embodied_cpu = embodied_emissions(cpu_embodied, duration_h, cpu_count, cpu_lifespan, cpu_cores)

        embodied_gpu = embodied_emissions(gpu_embodied, duration_h, gpu_count, gpu_lifespan, gpu_cores)

        embodied_ram = embodied_emissions(ram_embodied, duration_h, np.ceil(ram_used_gb), ram_lifespan, ram_gb)

        total_embodied = embodied_cpu + embodied_gpu + embodied_ram

        sci_total = sci_score(operational_emissions, total_embodied)

        print(f"SCI score: {sci_total}")

        embodied_formula_cpu = latex_embodied_emissions("CPU", cpu_embodied, duration_h, cpu_count, cpu_lifespan, cpu_cores, embodied_cpu)
        embodied_formula_gpu = latex_embodied_emissions("GPU", gpu_embodied, duration_h, gpu_count, gpu_lifespan, gpu_cores, embodied_gpu)
        embodied_formula_ram = latex_embodied_emissions("RAM", ram_embodied, duration_h, ram_used_gb, ram_lifespan, ram_gb, embodied_ram)

        latex_formula_sci = make_latex_formula(energy_consumed, region, total_embodied, sci_total)

        tex_code = f"""
\\textbf{{Cálculo de SCI para el experimento: {project_name}}}
\\begin{{equation}}
\\begin{{split}}
{embodied_formula_cpu} \\\\
{embodied_formula_gpu} \\\\
{embodied_formula_ram} \\\\
{latex_formula_sci}
\\end{{split}}
\\label{{eq:sci:{project_name}}}
\\end{{equation}}
"""

        print(embodied_formula_cpu)
        print(embodied_formula_gpu)
        print(embodied_formula_ram)
        print(latex_formula_sci)

        print("LaTeX code for the SCI calculation:")
        print(tex_code)

        print("--------------------------------\n\n")
        df_sci_scores = pd.concat([df_sci_scores, pd.DataFrame({"project_name": [project_name], "sci_score": [sci_total]})], ignore_index=True)

    df_sci_scores.to_csv("./analysis/sci_scores.csv", index=False)