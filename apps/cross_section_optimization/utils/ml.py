from typing import Iterable, Optional, overload

import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PyFuncModel
import pandas as pd
import torch

from .constants import CANONICAL_SCORE_NAME, INTERNAL_FORCE_COMPONENTS
from .torchutils import get_torch_device


def delete_all_logged_models(experiment_name: Optional[str] = None) -> None:
    """
    Delete all logged models from MLflow. If an experiment name is provided,
    only models associated with that experiment will be deleted.
    
    Parameters
    ----------
    experiment_name : Optional[str], optional
        The name of the experiment whose logged models should be deleted. If None,
        all logged models across all experiments will be deleted. Default is None.
    """
    
    client = MlflowClient()

    if experiment_name:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_ids = [experiment.experiment_id]
    else:
        experiment_ids = None
        
    logged_models = mlflow.search_logged_models(experiment_ids=experiment_ids)

    for m in logged_models.itertuples():
        client.delete_logged_model(m.model_id)
        

def get_all_runs(task:str, section_type:str) -> pd.DataFrame:
    """
    Retrieve all MLflow runs for a given task and section type.
    """
    # get the experiment
    experiment_name = f"{task}__{section_type}"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # get all runs for the experiment
    df_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
    )

    return df_runs


def get_all_logged_models(task:str, section_type:str) -> pd.DataFrame:
    """
    Retrieve all logged models from MLflow.
    """
    experiment_name = f"{task}__{section_type}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return mlflow.search_logged_models(experiment_ids=[experiment.experiment_id])


def get_section_variables(config: dict) -> list[str]:
    """Return the list of section variable names."""
    section_data: dict = config["section"]
    section_variables: list[str] = []     # list of variable parameters
    for p in section_data["params"]:
        if section_data["params"][p]["variable"]:
            section_variables.append(p)
    return section_variables


def get_predictor_columns(config: dict, task: str) -> list[str]:
    """Return the predictor column names for a given task."""
    section_variables = get_section_variables(config)
    if task == "failure_prediction":
        return section_variables + INTERNAL_FORCE_COMPONENTS
    elif task == "utilization_estimation":
        return section_variables + INTERNAL_FORCE_COMPONENTS
    elif task == "section_estimation":
        return section_variables
    elif task == "geometry_validation":
        return section_variables
    else:
        raise NotImplementedError(f"Task '{task}' is not implemented.")

    
def get_sample_data(
    config: dict,
    data_file_path: str,
    task: str,
    n_sample: int=100, 
    flavour: str="torch", 
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """Retrieve sample input data for a given task and flavour."""

    predictor_columns = get_predictor_columns(config, task)
    df = pd.read_csv(data_file_path)
    df = df.dropna()
    X_df = df[predictor_columns].iloc[:n_sample]
    
    if flavour == "sklearn":
        return X_df
    elif flavour == "torch":
        device = get_torch_device()
        X_np = X_df.to_numpy(dtype=np.float32)
        X_tensor = torch.tensor(X_np, device=device)
        return X_tensor


class Predictor:
    """A predictor class that wraps around a model and handles different flavours."""
    
    def __init__(self, model: PyFuncModel, predictor_columns: list[str], flavour:str="sklearn"):
        self.model = model
        self.predictor_columns = predictor_columns
        self.flavour = flavour

    def predict(self, X: pd.DataFrame | torch.Tensor | list) -> Iterable:
        """Make predictions using the model."""

        if isinstance(X, (list, np.ndarray)):
            X = np.array(X)
            X = np.atleast_2d(X)
            X = pd.DataFrame(X, columns=self.predictor_columns)
                
        if self.flavour == "sklearn":
            return self.model.predict(X[self.predictor_columns])
        elif self.flavour == "torch":
            if not isinstance(X, torch.Tensor):
                device = get_torch_device()
                X_np = X[self.predictor_columns].to_numpy(dtype=np.float32)
                X = torch.tensor(X_np, device=device)
            return self.model(X).detach().cpu().numpy().squeeze()
        else:
            raise NotImplementedError(f"Flavour '{self.flavour}' is not implemented.")
        
    def __str__(self):
        return self.model.__str__()
    
    def __repr__(self):
        return self.model.__repr__()


def get_best_model(
    config: dict,
    data_file_path: str,
    task:str, 
    section_type:str,
    *, 
    test:bool=False, 
    n_sample: int=5,
    flavour: Optional[str]=None
) -> Predictor:

    df_runs = get_all_runs(task, section_type)
    df_models = get_all_logged_models(task, section_type)

    df_models = df_models.merge(
        df_runs[["run_id", "tags.library", "tags.section_type", "tags.task", f"metrics.{CANONICAL_SCORE_NAME}"]],
        left_on="source_run_id",
        right_on="run_id",
        how="inner"
    ).drop(columns=["source_run_id"], axis=1)
    df_models.rename(
        columns={
            f"metrics.{CANONICAL_SCORE_NAME}": CANONICAL_SCORE_NAME,
            "tags.library": "flavour",
            "tags.section_type": "section_type",
            "tags.task": "task"
        }, 
        inplace=True
    )

    df_models = df_models[df_models["flavour"].isin(["sklearn", "torch"])]
    if flavour:
        df_models = df_models[df_models["flavour"] == flavour]
        if df_models.empty:
            raise ValueError(f"No models found for flavour '{flavour}'.")
        
    df_models.sort_values(CANONICAL_SCORE_NAME, ascending=False, inplace=True)
    df_best_model = df_models.iloc[0]
    best_model_run_id = df_best_model["run_id"]
    best_model_name = df_best_model["name"]
    best_model_uri = f"runs:/{best_model_run_id}/{best_model_name}"
    best_model_flavour = df_best_model["flavour"]
    
    if best_model_flavour == "sklearn":
        best_model = mlflow.sklearn.load_model(best_model_uri)
    elif best_model_flavour == "torch":
        device = get_torch_device()
        best_model = mlflow.pytorch.load_model(best_model_uri, device=device)
        best_model.eval()
    else:
        raise NotImplementedError(f"Library '{best_model_flavour}' is not implemented.")

    predictor_columns = get_predictor_columns(config, task)
    predictor = Predictor(best_model, predictor_columns=predictor_columns, flavour=best_model_flavour)

    if test:
        sample_data = get_sample_data(config, data_file_path, task, n_sample=n_sample, flavour=best_model_flavour)
        predictor.predict(sample_data)
    
    return predictor
        
        
def calculate_performance_metrics(confusion_matrix: np.ndarray) -> dict:
    """
    Compute classification performance metrics from a binary confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray of shape (2, 2)
        Confusion matrix in the form::

            [[TN, FP], 
             [FN, TP]]

        where:
        - TP: True Positives
        - FN: False Negatives
        - FP: False Positives
        - TN: True Negatives
        
    Returns
    -------
    dict
        A dictionary containing the following performance metrics:

        - **accuracy** : float  
          Proportion of correctly classified samples:  
          (TP + TN) / (TP + TN + FP + FN).

        - **precision** (PPV) : float  
          Positive Predictive Value. Fraction of predicted positives that are
          truly positive:  
          TP / (TP + FP).

        - **recall** (sensitivity) : float  
          True Positive Rate. Fraction of true positives correctly identified:  
          TP / (TP + FN).

        - **f1** : float  
          Harmonic mean of precision and recall, balancing the two:  
          2 * (precision * recall) / (precision + recall).

        - **specificity** : float  
          True Negative Rate. Fraction of true negatives correctly identified:  
          TN / (TN + FP).

        - **sensitivity** : float  
          Alias for recall. Fraction of true positives correctly identified.

        - **NPV** : float  
          Negative Predictive Value. Fraction of predicted negatives that are
          truly negative:  
          TN / (TN + FN).

        - **PPV** : float  
          Alias for precision. Fraction of predicted positives that are truly
          positive.

    Notes
    -----
    - All metrics are returned as floats between 0 and 1.
    - If a denominator is zero, the corresponding metric is set to 0 to avoid
      division by zero.
    """
    
    [[TN, FP], [FN, TP]] = confusion_matrix

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "NPV": npv,
        "PPV": ppv,
    }
    

@overload
def score_failure_predictor(y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
@overload
def score_failure_predictor(cm: np.ndarray) -> float: ...


def score_failure_predictor(
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    cm: np.ndarray | None = None,
) -> float:
    """
    Custom scoring function for evaluating failure prediction models.
    Can be called with either:
      - (y_true, y_pred)
      - (cm)
    """
    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("Either provide (y_true, y_pred) or cm.")
        cm = confusion_matrix(y_true, y_pred)

    metrics = calculate_performance_metrics(cm)
    score = 0.8 * metrics['NPV'] + 0.2 * metrics['PPV']
    return float(score)


def normalize_confusion_matrix(cm: np.ndarray, mode: str = "true") -> np.ndarray:
    """
    Normalize a confusion matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix to be normalized.
    mode : str, optional
        Normalization mode. Options are:
        - "true": Normalize by true labels (row-wise).
        - "pred": Normalize by predicted labels (column-wise).
        - "all": Normalize by total sum (global).
        Default is "true".
    """
    if mode == "true":      # row-wise
        return cm.astype(float) / cm.sum(axis=1, keepdims=True)
    elif mode == "pred":    # column-wise
        return cm.astype(float) / cm.sum(axis=0, keepdims=True)
    elif mode == "all":     # global
        return cm.astype(float) / cm.sum()
    else:
        raise ValueError("mode must be 'true', 'pred', or 'all'")
  
  
def canonical_regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the canonical regression score as the negative Mean Squared Error (MSE).
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
        
    Returns
    -------
    float
        The canonical regression score (negative MSE).
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    y_std = float(np.std(y_true))
    if y_std > 0:
        nrmse = rmse / y_std
    else:
        fallback_scale = max(1e-8, abs(float(np.mean(y_true))))
        nrmse = rmse / fallback_scale
    canonical_score = 1 / (1 + nrmse)
    return canonical_score
