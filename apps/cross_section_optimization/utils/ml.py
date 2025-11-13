from typing import Optional, overload

import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient


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
