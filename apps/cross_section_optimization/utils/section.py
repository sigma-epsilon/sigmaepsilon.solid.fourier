from sectionproperties.analysis import Section
from sectionproperties.pre.library import rectangular_hollow_section, i_section
from sectionproperties.pre import Material
from typing import Callable
from types import NoneType
import numpy as np
from .constants import INTERNAL_FORCE_COMPONENTS
from .logger import get_logger


logger = get_logger()

geometry_constructors = {
    "rectangular_hollow_section": rectangular_hollow_section,
    "i_section": i_section,
    # Add other handlers here as needed
}


def _get_geometry_constructor_by_name(name: str) -> Callable:
    """Returns the section handler function based on the given name."""
    
    if name not in geometry_constructors:
        raise ValueError(f"Handler '{name}' is not recognized.")
    return geometry_constructors[name]


def construct_section(
    geometry_constructor: Callable | str,
    params: dict,
    material: Material | dict,
    mesh_sizes: int | list[int] | NoneType = None,
    calculate: bool = True,
) -> Section:
    """Builds a section from a geometry constructor and parameters and optionally creates the mesh.
    
    Parameters
    ----------
    geometry_constructor : Callable | str
        The section handler function or its name as a string.
    params : dict
        The parameters to be passed to the geometry constructor.
    material : Material
        The material to be assigned to the section.
    mesh_sizes : int | list[int] | NoneType, optional
        The mesh sizes to be used for mesh generation. If None, no mesh is created.
    """
    if isinstance(geometry_constructor, str):
        
        geometry_constructor = _get_geometry_constructor_by_name(geometry_constructor)
    if isinstance(material, dict):
        material = Material(**material)
    
    geom = geometry_constructor(**params, material=material)
    
    if mesh_sizes:
        geom.create_mesh(mesh_sizes=mesh_sizes)

    section = Section(geometry=geom)
    
    if calculate:
        section.calculate_geometric_properties()
        section.calculate_warping_properties()

    return section


def utilization(section: Section, loads: dict) -> float:
    """
    Calculate the maximum utilization of a cross section under given loads.

    Parameters
    ----------
    section : Section
        The cross section to be analyzed.
    loads : dict
        The loads to be applied to the section.
        Keys (all optional, default 0.0):
            'n'   : Axial force
            'vx'  : Shear force in x
            'vy'  : Shear force in y
            'mxx' : Moment about x
            'myy' : Moment about y
            'm11' : Moment about principal axis 1
            'm22' : Moment about principal axis 2
            'mzz' : Torsional moment
    """
    stress = section.calculate_stress(**loads)
    material = section.materials[0]
    yield_strength = material.yield_strength
    sig_vm_max = np.max(stress.get_stress()[0]["sig_vm"])
    utilization_max = sig_vm_max / yield_strength
    return utilization_max


def section_properties(section: Section) -> dict:
    """Extract key geometric properties of the section."""
    area = section.get_area()
    ixx_c, iyy_c, ixy_c = section.get_eic()
    g_eff = section.get_g_eff()
    props = {
        "area": area,
        "ixx": ixx_c,
        "iyy": iyy_c,
        "ixy": ixy_c,
        "g_eff": g_eff,
    }
    return props


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