# Generate learning data for model training

# Import necessary libraries
from sectionproperties.analysis import Section
from sectionproperties.pre.library import rectangular_hollow_section
from sectionproperties.pre import Material
import numpy as np
import pandas as pd
import multiprocessing
import random
import json
import logging

logging.basicConfig(level=logging.DEBUG, force=True)

# Load configuration from JSON file
with open("config.json", "r") as f:
    config = json.load(f)

load_components = ["n", "mxx", "myy", "vx", "vy", "mzz"]

# this is gonna be replaced later
load_ranges = {k: (0, 1000) for k in load_components}

# Load material data and define the material
material_params = config["material"]
material = Material(**material_params)

# Load section data
section_data = config["section"]


def utilization(section_params:dict, loads:dict) -> float:
    """Calculate the maximum utilization of a cross section under given loads."""
    # Define the section
    geom_params = {k:v for k,v in section_params.items() if k in section_data["params"]}
    geom = rectangular_hollow_section(**geom_params, material=material)
    geom.create_mesh(mesh_sizes=section_data["mesh_sizes"])
    sec = Section(geometry=geom)

    # Calculate geometric properties
    sec.calculate_geometric_properties()
    sec.calculate_warping_properties()
    #sec.calculate_frame_properties()

    # Calculate maximum Von-Mises stress under given loads
    stress = sec.calculate_stress(**loads)
    sig_vm_max = np.max(stress.get_stress()[0]["sig_vm"])
    utilization_max = sig_vm_max / material_params["yield_strength"]
    return utilization_max


def default_section_params() -> dict:
    """Generate default cross section parameters for a rectangular hollow section."""
    params = {}
    for p in section_data["params"].keys():
        params[p] = section_data["params"][p]["default"]
    return params
    

def random_section_params() -> dict:
    """Generate random cross section parameters for a rectangular hollow section."""
    params = {}
    for p in section_data["params"].keys():
        if section_data["params"][p]["variable"]:
            min_value, max_value = section_data["params"][p]["range"]
            params[p] = random.uniform(min_value, max_value)
        else:
            params[p] = section_data["params"][p]["default"]
    return params


def random_loads() -> dict:
    """Generate a dictionary with random loads for section analysis."""
    loads = {}
    for k in load_components:
        loads[k] = random.uniform(load_ranges[k][0], load_ranges[k][1])
    return loads


def generate_sample(args: tuple[dict, dict]) -> dict:
    """Generate a single data sample of section parameters, loads, and utilization."""
    section_params, loads = args
    try:
        util = utilization(section_params, loads)
    except Exception as e:
        logging.error(f"Error calculating utilization for params {section_params} and loads {loads}: {e}")
        util = None
    result = {**section_params, **loads, "utilization": util}
    return result


def find_internal_force_limits(section_params: dict) -> dict:
    """Find the min and max load values for each load component that 
    lead to utilization of at least 1.0."""

    def _find_extreme_load_value(load_component:str, load_step:float) -> float:
        load_value = load_step
        utilization_value = 0.0
        while (utilization_value < 0.9) or (utilization_value > 1.3):
            # calculate utilization for current load value
            loads = {component: 0.0 for component in load_components}
            loads[load_component] = load_value
            new_utilization_value = utilization(section_params, loads)
            # calculate new step size based on linear prediction
            delta_u = new_utilization_value - utilization_value
            load_step = (1 - new_utilization_value) * load_step / delta_u if delta_u != 0 else load_step
            # update utilization value
            utilization_value = new_utilization_value
            # increment load value
            load_value += load_step
            logging.debug(f"Testing {load_component}={load_value:.2f}, Utilization={utilization_value:.4f}, Step={load_step:.2f}")
        return load_value
    
    logging.info("Finding internal force limits...")
    
    results = {component: None for component in load_components}
    for load_component in load_components:
        logging.info(f"Finding limits for load component: {load_component}")
        max_value = _find_extreme_load_value(load_component, load_step=1.0)
        min_value = _find_extreme_load_value(load_component, load_step=-1.0)
        results[load_component] = (min_value, max_value)
        logging.info(f"Found limits for load component {load_component}: {min_value}, {max_value}")

    logging.info("Finished finding internal force limits.")
    return results

if __name__ == "__main__":

    # print("Default Section Params:", default_section_params())
    # print("Default Section Params:", default_section_params())
    # print("Random Section Params:", random_section_params())
    # print("Random Loads:", random_loads())
    # print("Generated Sample:", generate_sample((random_section_params(), random_loads())))
    # print("Utilization:", utilization(default_section_params(), random_loads()))

    # Load ranges for random generation
    logging.info("Calculating load ranges based on default section parameters...")
    load_ranges = find_internal_force_limits(default_section_params())
    logging.info(f"Determined load ranges: {load_ranges}")

    num_sections = config["num_sections"]
    num_loads_per_section = config["num_load_cases_per_section"]
    logging.info("Generating random section and load parameters...")
    logging.debug(f"Number of sections: {num_sections}, Number of loads per section: {num_loads_per_section}")
    param_load_pairs = []
    for section_param_id in range(num_sections):
        section_params = random_section_params()
        section_params.update({"section_param_id": section_param_id})
        for _ in range(num_loads_per_section):
            loads = random_loads()
            param_load_pairs.append((section_params, loads))
    logging.info(f"Generated {len(param_load_pairs)} parameter-load pairs.")
    
    num_workers = 8
    logging.info(f"Generating data with {num_workers} parallel workers...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        data = pool.map(generate_sample, param_load_pairs)

    logging.info("Data generation completed. Saving to CSV...")
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)
    logging.info("Data saved to data.csv.")
