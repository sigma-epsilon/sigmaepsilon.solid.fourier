# Cross Section Optimization with SigmaEpsilon, Scikit-Learn and PyTorch

In this application we

- train a ML model to learn a cross section ([train_model.ipynb](train_model.ipynb))
- optimize the cross section of a simply supported beam using a genetic algorithm ([optimize.ipynb](optimize.ipynb))

## Installation

To install the dependencies for this app:

```console
cd apps/cross_section_optimization
pip install -r requirements.txt
```

## Usage

```console
python '1 - generate_learning_data.py' --config config.json --loglevel DEBUG --num_sections 2 --num_load_cases_per_section 2 --num_workers 2 --output out.csv
```
