import numpy as np


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
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "NPV": npv,
        "PPV": ppv,
    }