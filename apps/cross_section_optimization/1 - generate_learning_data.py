# Generate learning data for model training

from typing import Any
from sectionproperties.analysis import Section
import pandas as pd
import multiprocessing
import random
import json
import argparse
from utils import INTERNAL_FORCE_COMPONENTS
from utils.section import construct_section, utilization, section_properties
from utils.logger import get_logger, set_log_level
from tqdm import tqdm

logger = get_logger()


def default_section_params(section_data: dict) -> dict:
    """Generate default cross section parameters for a rectangular hollow section."""
    params = {}
    for p in section_data["params"].keys():
        params[p] = section_data["params"][p]["default"]
    return params
    

def random_section_params(section_data: dict) -> dict:
    """Generate random cross section parameters for a rectangular hollow section."""
    params = {}
    for p in section_data["params"].keys():
        if section_data["params"][p]["variable"]:
            min_value, max_value = section_data["params"][p]["range"]
            params[p] = random.uniform(min_value, max_value)
        else:
            params[p] = section_data["params"][p]["default"]
    return params


def random_loads(load_ranges: dict) -> dict:
    """Generate a dictionary with random loads for section analysis."""
    loads = {}
    for k in INTERNAL_FORCE_COMPONENTS:
        loads[k] = random.uniform(load_ranges[k][0], load_ranges[k][1])
    return loads


def generate_sample(args: tuple[dict, dict]) -> dict:
    """Generate a single data sample of section parameters, loads, and utilization."""
    (
        section_param_id,
        section_type, 
        section_params, 
        mesh_sizes, 
        material_params, 
        loads
    ) = args
    try:
        section = construct_section(
            geometry_constructor=section_type,
            params=section_params,
            material=material_params,
            mesh_sizes=mesh_sizes,
            calculate=True,
        )
        util = utilization(section, loads)
        stiffness_props = section_properties(section)
    except Exception as e:
        if "TopologyException" in str(e):
            logger.warning(f"TopologyException for params {section_params} and loads {loads}: {e}")
        else:
            logger.error(f"Error generating sample for params {section_params} and loads {loads}: {e}")
        util = None
        stiffness_props = {}
        
    result = {
        **section_params, 
        **loads,
        **stiffness_props,
        "utilization": util,
        "section_param_id": section_param_id,
        "section_type": section_type,
    }
    return result


def find_internal_force_limits(section: Section) -> dict:
    """Find the min and max load values for each load component that 
    lead to utilization of at least 1.0."""

    def _find_extreme_load_value(load_component:str, load_step:float) -> float:
        load_value = load_step
        utilization_value = 0.0
        while (utilization_value < 0.9) or (utilization_value > 1.3):
            # calculate utilization for current load value
            loads = {component: 0.0 for component in INTERNAL_FORCE_COMPONENTS}
            loads[load_component] = load_value
            new_utilization_value = utilization(section, loads)
            # calculate new step size based on linear prediction
            delta_u = new_utilization_value - utilization_value
            load_step = (1 - new_utilization_value) * load_step / delta_u if delta_u != 0 else load_step
            # update utilization value
            utilization_value = new_utilization_value
            # increment load value
            load_value += load_step
            logger.debug(f"Testing {load_component}={load_value:.2f}, Utilization={utilization_value:.4f}, Step={load_step:.2f}")
        return load_value
    
    logger.info("Finding internal force limits...")
    
    results = {component: None for component in INTERNAL_FORCE_COMPONENTS}
    for load_component in INTERNAL_FORCE_COMPONENTS:
        logger.info(f"Finding limits for load component: {load_component}")
        max_value = _find_extreme_load_value(load_component, load_step=1.0)
        min_value = _find_extreme_load_value(load_component, load_step=-1.0)
        results[load_component] = (min_value, max_value)
        logger.info(f"Found limits for load component {load_component}: {min_value}, {max_value}")

    logger.info("Finished finding internal force limits.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross Section Optimization")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Set logger level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--num_sections", type=int, default=-1, help="Number of sections to generate")
    parser.add_argument("--num_load_cases_per_section", type=int, default=-1, help="Number of load cases per section")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers for data generation")
    parser.add_argument("--output", type=str, default="data.csv", help="Output CSV file name")
    args = parser.parse_args()

    set_log_level(args.loglevel)

    # Load configuration from JSON file
    with open(args.config, "r") as f:
        config = json.load(f)

    ## TESTS - uncomment to run
    # print("Default Section Params:", default_section_params())
    # print("Default Section Params:", default_section_params())
    # print("Random Section Params:", random_section_params())
    # print("Random Loads:", random_loads())
    # print("Generated Sample:", generate_sample((random_section_params(), random_loads())))
    # print("Utilization:", utilization(default_section_params(), random_loads()))
    
    material_params = config["material"]
    section_data = config["section"]
    section_type = section_data["type"]
    section = construct_section(
        geometry_constructor=section_type,
        params=default_section_params(section_data),
        material=material_params,
        mesh_sizes=section_data["mesh_sizes"],
        calculate=True,
    )
    
    # Load ranges for random generation
    logger.info("Calculating load ranges based on default section parameters...")
    load_ranges = find_internal_force_limits(section)
    logger.info(f"Determined load ranges: {load_ranges}")

    if (num_sections := args.num_sections) < 0:
        num_sections = config["num_sections"]
    if (num_loads_per_section := args.num_load_cases_per_section) < 0:
        num_loads_per_section = config["num_load_cases_per_section"]
        
    assert num_sections > 0, "Number of sections must be positive."
    assert num_loads_per_section > 0, "Number of load cases per section must be positive."
        
    logger.info("Generating random section and load parameters...")
    logger.debug(f"Number of sections: {num_sections}, Number of loads per section: {num_loads_per_section}")
    tasks = []
    for section_param_id in range(num_sections):
        section_params = random_section_params(section_data)
        mesh_sizes = section_data["mesh_sizes"]
        for _ in range(num_loads_per_section):
            loads = random_loads(load_ranges)
            tasks.append((section_param_id, section_type, section_params, mesh_sizes, material_params, loads))
    logger.info(f"Generated {len(tasks)} tasks.")

    num_workers = args.num_workers
    logger.info(f"Generating data with {num_workers} parallel workers...")

    data = []
    with multiprocessing.Pool(num_workers) as pool:
        with tqdm(total=len(tasks), desc="Generating samples") as pbar:

            def on_success(result: Any) -> None:
                """Called after each successful worker execution."""
                data.append(result)
                pbar.update(1)

            def on_error(e: Exception) -> None:
                """Called after a failed worker execution."""
                logger.error(f"Error in worker: {e}")
                pbar.update(1)

            for t in tasks:
                pool.apply_async(
                    generate_sample,
                    args=(t,),
                    callback=on_success,
                    error_callback=on_error
                )

            pool.close()
            pool.join()

    logger.info(f"Data generation completed. Saving results to {args.output} ...")
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    logger.info(f"Saved results to {args.output}.")
