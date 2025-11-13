from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
import numpy as np

__all__ = ["canonical_regression_score_from_batches"]


@dataclass
class _RunningRegStatsMulti:
    """
    Running statistics for multi-target regression canonical score computation.
    Uses Welford's method for numerically stable online updates of mean and variance.
    """
    n: int = 0                             # number of valid samples accumulated
    mean_y: Optional[torch.Tensor] = None  # (D,) running mean of y_true
    M2_y: Optional[torch.Tensor] = None    # (D,) running sum of squares for variance
    sse: Optional[torch.Tensor] = None     # (D,) sum of squared errors per target

    def _init_if_needed(self, D: int, device: torch.device = torch.device("cpu")) -> None:
        if self.mean_y is None:
            self.mean_y = torch.zeros(D, dtype=torch.float64, device=device)
            self.M2_y   = torch.zeros(D, dtype=torch.float64, device=device)
            self.sse    = torch.zeros(D, dtype=torch.float64, device=device)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        # Ensure 2D shapes: (N, D)
        y_true = y_true.detach().to("cpu")
        y_pred = y_pred.detach().to("cpu")

        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(-1)

        if y_true.ndim != 2 or y_pred.ndim != 2:
            raise ValueError(f"Expected y_true/y_pred to be 1D or 2D; got {y_true.shape} and {y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Pred/target shape mismatch: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}")

        N, D = y_true.shape
        self._init_if_needed(D)

        # Mask out any rows that contain non-finite values in either y_true or y_pred (per sample)
        finite_true = torch.isfinite(y_true)
        finite_pred = torch.isfinite(y_pred)
        row_mask = torch.logical_and(finite_true, finite_pred).all(dim=1)
        if not torch.any(row_mask):
            return

        y_true = y_true[row_mask]
        y_pred = y_pred[row_mask]

        # Residuals and SSE per target
        resid = (y_pred - y_true).to(torch.float64)
        batch_sse = torch.sum(resid * resid, dim=0)  # (D,)

        # Welford parallel update per target for y_true
        b_n = y_true.shape[0]
        b_mean = torch.mean(y_true.to(torch.float64), dim=0)       # (D,)
        b_M2   = torch.sum((y_true.to(torch.float64) - b_mean) ** 2, dim=0)  # (D,)

        if self.n == 0:
            self.n      = b_n
            self.mean_y = b_mean
            self.M2_y   = b_M2
            self.sse    = batch_sse
        else:
            delta  = b_mean - self.mean_y                      # (D,)
            new_n  = self.n + b_n
            # Update running mean
            self.mean_y = self.mean_y + delta * (b_n / new_n)
            # Update running M2 (parallel Welford)
            self.M2_y = self.M2_y + b_M2 + (delta * delta) * (self.n * b_n / new_n)
            # SSE
            self.sse = self.sse + batch_sse
            # Count
            self.n = new_n

    def finalize_score(self) -> float:
        """
        Returns a single scalar: the mean of per-target canonical scores.
        Canonical score per target d: 1 / (1 + NRMSE_d)
        where NRMSE_d = RMSE_d / std(y_true_d).
        """
        if self.n == 0:
            return float("nan")

        # Per-target RMSE and std (population, matching numpy default in original code)
        rmse  = torch.sqrt(self.sse / self.n)                 # (D,)
        y_var = self.M2_y / self.n                            # (D,)
        y_std = torch.sqrt(y_var)                             # (D,)

        # Fallback scale for zero-variance targets: max(1e-8, |mean_y|)
        fallback_scale = torch.clamp(self.mean_y.abs(), min=1e-8)  # (D,)
        denom = torch.where(y_std > 0, y_std, fallback_scale)      # (D,)

        nrmse  = rmse / denom                                  # (D,)
        scores = 1.0 / (1.0 + nrmse)                           # (D,)
        overall = scores.mean().item()
        return float(overall)


@torch.no_grad()
def canonical_regression_score_from_batches(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    output_key: Optional[str] = None,
    target_key: Optional[str] = None
) -> float:
    """
    Compute canonical regression score = mean_d [ 1 / (1 + NRMSE_d) ] over a DataLoader, in a single pass.

    - Supports single-target (D=1) and multi-target (D>1) regression.
    - Works with batches that are (inputs, targets) or dicts containing inputs/targets.
    - Accepts model outputs of shape (N,), (N,1), or (N,D).

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch regression model.
    loader : DataLoader
        Yields (x, y) or a dict with inputs/targets.
    device : Optional[torch.device]
        Device for inference. If None, inferred from model params (falls back to CPU).
    output_key : Optional[str]
        If each batch is a dict and the model expects a dict, you can pull model inputs via this key.
        Usually leave None; we try common patterns automatically.
    target_key : Optional[str]
        If batch is a dict, the key for targets (tries 'targets'/'y'/'labels' if None).

    Returns
    -------
    float
        Canonical regression score in (0, 1], higher is better.
        For multi-target regression, this is the average of per-target canonical scores.
    """
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    stats = _RunningRegStatsMulti()

    for batch in loader:
        # --- unpack batch (tuple/list or dict) ---
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            # Inputs: try explicit key if provided, else common names
            if output_key is not None:
                inputs = batch[output_key]
            else:
                inputs = batch.get("inputs", batch.get("x", batch.get("features")))
            # Targets: explicit or common names
            if target_key is not None:
                targets = batch[target_key]
            else:
                targets = batch.get("targets", batch.get("y", batch.get("labels")))
            if inputs is None or targets is None:
                raise ValueError("Could not infer 'inputs' and 'targets' from batch dict.")
        else:
            raise ValueError("DataLoader must yield (inputs, targets) or a dict containing them.")

        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Ensure 2D regression output: (N, D)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(-1)
        elif outputs.ndim == 2 and outputs.shape[-1] == 1:
            # already (N,1)
            pass
        elif outputs.ndim != 2:
            raise ValueError(f"Expected regression output of shape (N,), (N,1), or (N,D); got {tuple(outputs.shape)}")

        # Ensure targets are (N, D)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        elif targets.ndim == 2 and targets.shape[-1] >= 1:
            pass
        else:
            raise ValueError(f"Expected targets of shape (N,), (N,1), or (N,D); got {tuple(targets.shape)}")

        if outputs.shape != targets.shape:
            raise ValueError(f"Pred/target shape mismatch: {tuple(outputs.shape)} vs {tuple(targets.shape)}")

        # Update running stats on CPU
        stats.update(y_true=targets, y_pred=outputs)

    return stats.finalize_score()


def confusion_matrix_from_predictions(targets: torch.Tensor, predictions: torch.Tensor) -> np.array:
    """
    Compute the confusion matrix from the model's predictions.
    
    Parameters
    ----------
    targets : torch.Tensor
        True binary labels (0 or 1).
    predictions : torch.Tensor
        Predicted binary labels (0 or 1).
    
    Returns
    -------
    np.array
        Confusion matrix as a 2x2 numpy array:
        [[TN, FP],
         [FN, TP]]
    """
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    return np.array([[tn, fp], [fn, tp]])


@torch.no_grad()
def collect_confusion_matrix_from_batches(
    model: nn.Module, 
    loader: DataLoader,
    *,
    threshold: Optional[float] = None, 
    temperature: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> np.array:
    """
    Collects the confusion matrix from the model's predictions on the given data loader.
    
    Parameters
    ----------
    model : nn.Module
        The trained model to evaluate.
    loader : DataLoader
        DataLoader providing the evaluation data.
    threshold : float, default=0.5
        Decision threshold for classifying positive vs negative.
    temperature : float, default=1.0
        Temperature scaling factor applied to logits before computing probabilities.
        
    Returns
    -------
    np.array
        Confusion matrix as a 2x2 numpy array:
        [[TN, FP],
         [FN, TP]]
    """
    model.eval()
    cm = np.zeros((2, 2), dtype=int)
    
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
            
    temperature = getattr(model, 'temperature', torch.tensor(1.0, device=device)) if temperature is None else temperature
    if isinstance(temperature, float):
        temperature = torch.tensor(temperature, device=device)

    threshold = getattr(model, 'threshold', torch.tensor(0.5, device=device)) if threshold is None else threshold
    if isinstance(threshold, float):
        threshold = torch.tensor(threshold, device=device)

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)#.squeeze(1)
        logits /= temperature.clamp(min=1e-6)

        # Compute probabilities and predictions
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).long()

        # Update confusion counts
        cm += confusion_matrix_from_predictions(targets, predictions)
        
    # Confusion matrix
    return cm