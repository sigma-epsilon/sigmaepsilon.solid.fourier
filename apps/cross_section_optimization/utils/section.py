from sectionproperties.analysis import Section
from sectionproperties.pre.library import rectangular_hollow_section, i_section
from sectionproperties.pre import Material
from typing import Callable
from types import NoneType
import numpy as np


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
