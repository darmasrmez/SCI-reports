import numpy as np

FESEN = 444 # Region-specific carbon intensity for Mexico(gCO2eq/kWh).
MI210_TE = 210 # Total embodied emissions for MI210(kgCO2eq).
EPYC_TE = 16 # Total embodied emissions for EPYC(kgCO2eq).
A100_TE = 16 # Total embodied emissions for A100(kgCO2eq).
RAM_TE = 0


def operational_energy(electricity_consumption, region_factor):
    """
    Calculate the operational energy for a given electricity consumption and region factor.
    Args:
        electricity_consumption: energy consumed by a software system for a functional unit of work, kilowatt hours (kWh).
        region_factor: The region factor of the system, Units: this shall be in grams of carbon per kilowatt hours (gCO2eq/kWh).
    Returns:
        The operational energy of the system.
    """
    return electricity_consumption * region_factor

def embodied_emissions(total_embodied:float, time_reserved:float, resource_reserved:float, expected_lifespan:float, total_resources:float):
    """
    Calculate the embodied emissions for a given total embodied, time reserved, resource reserved, expected lifespan, and total resources.
    Args:
        total_embodied: The total embodied emissions of the system, Units: this shall be in grams of carbon per kilowatt hours (gCO2eq/kWh).
        time_reserved: The time reserved for the system, Units: this shall be in years.
        resource_reserved: The resource reserved for the system, Units: this shall be in kilowatt hours (kWh).
        expected_lifespan: The expected lifespan of the system, Units: this shall be in hours.
        total_resources: The total resource of the system, Units: this shall be in kilowatt hours (kWh).
    Returns:
        The embodied emissions of the system.
    """
    time_share = time_reserved / expected_lifespan
    resource_share = resource_reserved / total_resources
    return total_embodied * time_share * resource_share

def sci_score(operational, embodied):
    return operational + embodied





if __name__ == "__main__":
    
