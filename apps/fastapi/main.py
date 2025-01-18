# main.py
import os
import ast
import json
import base64
import numpy as np

from fastapi import FastAPI
from typing import Dict, Any

# sigmaepsilon imports (install them with pip if not already installed)
from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.solid.material import (
    MindlinPlateSection as Section,
    ElasticityTensor,
    LinearElasticMaterial,
    HuberMisesHenckyFailureCriterion_SP,
)
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix
from sigmaepsilon.solid.fourier import NavierPlate, LoadGroup, PointLoad
from sigmaepsilon.mesh import PointData

app = FastAPI()

@app.post("/calculate_plate")
def calculate_plate(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Endpoint that runs a Mindlin plate analysis based on the given job_data.
    It returns:
        - The updated job_data (metadata).
        - Base64-encoded contents of two Parquet files (2D and 3D results).
    """

    # Wrap job_data in DeepDict for convenient nested access
    job_data = DeepDict.wrap(job_data)
    job_data.lock()

    # ------------------- 1) Extract parameters from job_data -------------------
    job_id = job_data["_meta", "job_id"]

    # Geometry
    length_X = job_data["geometry", "width"]
    length_Y = job_data["geometry", "height"]
    thickness = job_data["geometry", "thickness"]

    # Material properties
    young_modulus = job_data["material", "youngs_modulus"]
    poisson_ratio = job_data["material", "poissons_ratio"]
    yield_strength = job_data["material", "yield_strength"]

    # Solution parameters
    number_of_modes_X = job_data["calculation", "nx"]
    number_of_modes_Y = job_data["calculation", "ny"]

    # Load info
    load_position = job_data["load", "position"]
    load_value = job_data["load", "value"]

    # Convert string inputs to numeric if needed
    if isinstance(load_position, str):
        load_position = ast.literal_eval(load_position)
    if isinstance(load_value, str):
        load_value = ast.literal_eval(load_value)

    load_position = np.array(load_position, dtype=float)
    load_value = np.array(load_value, dtype=float)

    # ------------------- 2) Set up loads, material, section --------------------
    loads = LoadGroup(LC=PointLoad(load_position, load_value))

    # Hooke's law, stiffness, material
    hooke = elastic_stiffness_matrix(E=young_modulus, NU=poisson_ratio)
    frame = ReferenceFrame(dim=3)
    stiffness = ElasticityTensor(hooke, frame=frame, tensorial=False)
    failure_model = HuberMisesHenckyFailureCriterion_SP(yield_strength=yield_strength)
    material = LinearElasticMaterial(stiffness=stiffness, failure_model=failure_model)

    # Plate section stiffness
    section = Section(layers=[Section.Layer(material=material, thickness=thickness)])
    ABDS_matrix = section.elastic_stiffness_matrix()
    bending_stiffness = np.ascontiguousarray(ABDS_matrix[:3, :3])
    shear_stiffness   = np.ascontiguousarray(ABDS_matrix[3:, 3:])

    # ------------------- 3) Create mesh of evaluation points -------------------
    nx, ny = (150, 200)  # Example resolution (you can tweak these)
    x = np.linspace(0, length_X, nx)
    y = np.linspace(0, length_Y, ny)
    xv, yv = np.meshgrid(x, y)
    evaluation_points = np.stack((xv.flatten(), yv.flatten()), axis=1)

    # ------------------- 4) Solve the plate model ------------------------------
    plate = NavierPlate(
        (length_X, length_Y),
        (number_of_modes_X, number_of_modes_Y),
        D=bending_stiffness,
        S=shear_stiffness,
    )
    results = plate.linear_static_analysis(evaluation_points, loads)

    # ------------------- 5) Gather 2D results -----------------------------------
    load_case_results = results["LC"].values

    # ------------------- 6) Calculate 3D utilization results --------------------
    strains = results["LC"].strains
    z = np.linspace(-thickness / 2, thickness / 2, 20)
    rng = (-thickness / 2, thickness / 2)
    util, util_coords = section.utilization(
        strains=strains, rng=rng, z=z, coords=evaluation_points, return_coords=True
    )
    num_XY, num_Z = util_coords.shape[:2]
    util_coords = util_coords.reshape((num_XY * num_Z, 3))
    util_values = util.values.flatten()

    # ------------------- 7) Save results to Parquet files -----------------------
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    res2d_file_path = os.path.join(output_dir, f"{job_id}_res2d.parquet")
    res3d_file_path = os.path.join(output_dir, f"{job_id}_res3d.parquet")

    # 2D results
    pd_2d = PointData(evaluation_points, results=load_case_results)
    pd_2d.to_parquet(res2d_file_path)

    # 3D results
    pd_3d = PointData(util_coords, util=util_values)
    pd_3d.to_parquet(res3d_file_path)

    # ------------------- 8) Read & Base64-encode the Parquet files --------------
    with open(res2d_file_path, "rb") as f2d:
        res2d_encoded = base64.b64encode(f2d.read()).decode("utf-8")

    with open(res3d_file_path, "rb") as f3d:
        res3d_encoded = base64.b64encode(f3d.read()).decode("utf-8")

    # ------------------- 9) Update job_data and return --------------------------
    job_data.unlock()
    job_data["output"]["absolute_file_path_res2d"] = os.path.abspath(res2d_file_path)
    job_data["output"]["absolute_file_path_res3d"] = os.path.abspath(res3d_file_path)

    return {
        "metadata": job_data,
        "parquet_files": {
            "res2d": {
                "filename": f"{job_id}_res2d.parquet",
                "base64_data": res2d_encoded
            },
            "res3d": {
                "filename": f"{job_id}_res3d.parquet",
                "base64_data": res3d_encoded
            }
        }
    }
