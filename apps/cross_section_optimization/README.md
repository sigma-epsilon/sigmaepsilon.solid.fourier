# Cross Section Optimization with SigmaEpsilon, Scikit-Learn and PyTorch

In this application we

- train various ML model to predict the properties and behaviour of steel cross sections
- optimize the cross section of a simply supported beam using a genetic algorithm

Tech stack

- SigmaEpsilon for calculating the response of the beam and for optimization
- Scikit-Learn to train ML models for regression and classification
- PyTorch to train neural networks

See the file `requirements.txt` for the full list of dependencies.

## Installation

To install the dependencies for this app:

```console
cd apps/cross_section_optimization
pip install -r requirements.txt
```

## Usage

### Generate training data

To generate training data for a specific cross section and loads, call the file `1 - generate_learning_data.py` with suitable arguments. The following call creates 4000 data points using 8 workers and saves them as `out.csv`.

```console
python '1 - generate_learning_data.py' --config config.json --loglevel DEBUG --num_sections 200 --num_load_cases_per_section 20 --num_workers 8 --output out.csv
```

### Training models

[Documenting in progress...]

### Optimization

[Documenting in progress...]