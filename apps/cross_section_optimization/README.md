# Optimization of Beam Cross Sections in Python Using Machine-Learning Models

[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/benceeokf)

## Overview

The project demonstrates how to

- train various ML models to predict the properties and behaviour of steel cross sections
- select the best ML model for each task
- optimize the cross section of a simply supported beam using the best ML models and a genetic algorithm

### Reasons why the project might be of your interest

If you are a civil/structural engineer:

- [Documenting in progress...]

If you are a Data/ML Scientist:

- There are a few challenges the solution for which might be interesting for you. The Chan/Welford parallel variance update to construct a dataset invariant canonical regression metric could be one of these.
- There are many useful snippets you can take with yourself, see the previous point for a good example.

### Capabilities and limitations

[Documenting in progress...]

### Rationale of the solution strategy

Optimizing the cross section of a beam for multiple load cases involves quite some numbercrunching. A naive approach would be just simply put everything into the objective function and unleash a genetic algorithm. The main issue with this is that the genetic algorithm requires lots of evaulations until it finds the optimal solution. Just to calculate the properties of a cross section

- You have to discretize the domain of the cross section, which on its own involves the solution of a constrained optimization problem with possible thousands of unknowns, depending on how tough the geometry is.
- You have to solve a PDE (partial differential equation) on that domain with Neumann-type boundary conditions, which is far from trivial.

Depending on how challenging the geometry of the section is, these calculations can take quite some time. And all this is before we had calculated the response of the beam for the given loads, which then of course involves solving another PDE. And we are only talking about the simplest of all structures, a simply supported beam, not to mention real life structures.

Instead of this, the strategy here is to train machine learning models to

- tell if the parameters of a cross section define a valid geometry or not
- predict wether a cross-section fails under a given set of internal forces
- estimate the utilization of a cross-section for a given set of internal forces
- estimate the properties of the cross-section

and then to infer these models in the objective function. What we are not goig to do is to teach a ML model to calculate the response of the structure. It would be doable for the present case, but in general it's not a good idea due to the variety of the problem descriptions and boundary conditions.

### Prerequisites

There are many ways to interact with this project, depending what your goal is with it. For this reason, I list the minimum prerequisites separately for each level.

Prerequisites to use the solution:

- A basic understanding of the Python language.
- The ability to follow instructions on installing dependencies.
- The ability to follow instructions on executing commands in the terminal.

Prerequisites to extend the capabilities of the solution:

- Everything above.
- Intermediate Python.
- Domain knowledge in civil engineering, solid mechanics, numerical solution of PDEs and Machine Learning.

### Cornerstones of the tech stack

- [SigmaEpsilon.Solid.Fourier](https://sigmaepsilonsolidfourier.readthedocs.io/en/latest/index.html) for calculating the response of the beam.
- [SigmaEpsilon.Math](https://sigmaepsilonmath.readthedocs.io/en/latest/index.html#) for optimization
- [SectionProperties](https://sectionproperties.readthedocs.io/en/stable/index.html) to calculate properties of steel cross sections.
- [Scikit-Learn](https://scikit-learn.org/stable/index.html) to train ML models for regression and classification.
- [PyTorch](https://pytorch.org/) to build and train neural networks.
- [Papermill](https://papermill.readthedocs.io/en/latest/) to orchestrate the execution of parametric Notebooks.
- [MLflow](https://mlflow.org/docs/latest/ml/) to manage the ML lifecycle.

Of course, this list is not exhaustive, but these are the direct dependencies that would be very hard to replace if they didn't already exist. See the file `requirements.txt` for a full list of dependencies.

## Installation

To install the dependencies for this app:

```console
cd apps/cross_section_optimization
pip install -r requirements.txt
```

## Usage

### Step 0 - Describe the problem → `config.json`

Before you pull the triggers, you have to create a configuration file that describes your problem. This is where you select the type of cross section you want to optimize, set ranges for its variables, define loads, materials, etc. All later steps will feed on the information you provide here.

The configuration file is a JSON file (a text file with `.json` extension). The following snippet shows an example that describes a problem with an RHS section.

```json
{
    "material": {
        "name": "Steel",
        "elastic_modulus": 200000,
        "poissons_ratio": 0.3,
        "density": 7.85e-6,
        "yield_strength": 500,
        "color": "grey"
    },
    "section": {
        "type": "rectangular_hollow_section",
        "mesh_sizes": [4],
        "params": {
            "d": {
                "range": [20, 300],
                "default": 100,
                "variable": true
            },
            "b": {
                "range": [20, 300],
                "default": 100,
                "variable": true
            },
            "t": {
                "range": [2, 20],
                "default": 6,
                "variable": true
            },
            "r_out": {
                "range": [8, 20],
                "default": 15,
                "variable": true
            },
            "n_r": {
                "default": 4,
                "variable": false
            }
        }
    }
}
```

Save a file like this somewhere and note the path to it. The name of the file doesn't matter as long as it has the `.json` extension and it containts all required data.

### Step 1 - Generate training data

To generate training data for a specific cross section and loads, call the file `1 - generate_learning_data.py` with suitable arguments. The following call creates 4000 data points using 8 workers and saves them as `out.csv`.

```console
python '1 - generate_learning_data.py' --config config.json --loglevel DEBUG --num_sections 200 --num_load_cases_per_section 20 --num_workers 8 --output out.csv
```

Issuing the following command in the terminal would generate 50000 data points.

```console
cross_section_optimization % poetry run python '1 - generate_learning_data.py' --config config_rhs.json --loglevel DEBUG --num_sections 500 --num_load_cases_per_section 100 --num_workers 8 --output data_50000.csv
```

Note down the name of the configuration file and the generated csv file, you'll need these in later steps.

### Step 2 - Train models

[Documenting in progress...]

```python
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Step 3 - Optimize

[Documenting in progress...]

### Step 4 - Play and have fun

[Documenting in progress...]

## Where to go from here?

[Documenting in progress...]
