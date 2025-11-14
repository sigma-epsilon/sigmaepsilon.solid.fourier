from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.solid.fourier import LoadGroup, PointLoad, LineLoad
import random
from .constants import INTERNAL_FORCE_COMPONENTS

__all__ = ["get_loads_from_config", "random_loads"]


def get_loads_from_config(config: dict) -> LoadGroup:
    """
    Returns an instance of `LoadGroup` from the `sigmaepsilon.solid.fourier` library, based on the 
    provided configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing load definitions. The dictionary should have a key "loads" which contains
        a list of load configurations. Each load configuration should specify the type of load and its parameters.
    """
    loads_as_dict: DeepDict = DeepDict.wrap(config["loads"])

    beam_loads = LoadGroup()
    for load_config in loads_as_dict.containers(deep=True, dtype=dict):

        if not "type" in load_config:
            continue

        addr = load_config.address

        if load_config["type"] == "PointLoad":
            beam_loads[addr] = PointLoad(**load_config["params"])
        elif load_config["type"] == "LineLoad":
            beam_loads[addr] = LineLoad(**load_config["params"])
        else:
            raise NotImplementedError(f"Load type '{load_config['type']}' is not implemented.")

    return beam_loads


def random_loads(load_ranges: dict) -> dict:
    """Generate a dictionary with random loads for section analysis."""
    loads = {}
    for k in INTERNAL_FORCE_COMPONENTS:
        loads[k] = random.uniform(load_ranges[k][0], load_ranges[k][1])
    return loads