from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass


__all__ = ["canonical_regression_score_from_batches",]


@dataclass
class _RunningRegStats:
    n: int = 0                    # total samples
    mean_y: float = 0.0           # running mean of y_true
    M2_y: float = 0.0             # running sum of squares for variance
    sse: float = 0.0              # sum of squared errors: sum[(y_pred - y_true)^2]

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        # flatten & to CPU for stable numpy-like ops
        y_true = y_true.detach().reshape(-1).to("cpu")
        y_pred = y_pred.detach().reshape(-1).to("cpu")

        # mask non-finite values
        mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
        if not torch.any(mask):
            return
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # accumulate SSE
        resid = (y_pred - y_true)
        self.sse += float(torch.dot(resid, resid))

        # Chan/Welford parallel variance update for y_true
        b_n = y_true.numel()
        b_mean = float(torch.mean(y_true))
        # within-batch sum of squared deviations
        b_M2 = float(torch.sum((y_true - b_mean) ** 2))

        if self.n == 0:
            self.n = b_n
            self.mean_y = b_mean
            self.M2_y = b_M2
        else:
            delta = b_mean - self.mean_y
            new_n = self.n + b_n
            self.mean_y += delta * (b_n / new_n)
            self.M2_y += b_M2 + (delta * delta) * (self.n * b_n / new_n)
            self.n = new_n

    def finalize_score(self) -> float:
        if self.n == 0:
            return float("nan")
        rmse = np.sqrt(self.sse / self.n)
        y_var = self.M2_y / self.n            # population variance (np.std default)
        y_std = np.sqrt(y_var)
        if y_std > 0:
            nrmse = rmse / y_std
        else:
            fallback_scale = max(1e-8, abs(self.mean_y))
            nrmse = rmse / fallback_scale
        return float(1.0 / (1.0 + nrmse))


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
    Compute canonical regression score = 1 / (1 + NRMSE) over a DataLoader, in a single pass.

    - Works with batches that are (inputs, targets) or dicts containing inputs/targets.
    - Accepts model outputs of shape (N,) or (N, 1) for regression.

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
        If batch is a dict, the key for targets (tries 'targets'/'y' if None).

    Returns
    -------
    float
        Canonical regression score in (0, 1], higher is better.
    """
    model.eval()
    
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    stats = _RunningRegStats()

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

        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward
        outputs = model(inputs)

        # ensure 1D regression output
        if outputs.ndim > 1 and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
        elif outputs.ndim != 1:
            raise ValueError(f"Expected regression output of shape (N,) or (N,1); got {tuple(outputs.shape)}")

        # match targets shape
        if targets.ndim > 1 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        elif targets.ndim != 1:
            raise ValueError(f"Expected targets of shape (N,) or (N,1); got {tuple(targets.shape)}")

        if outputs.shape != targets.shape:
            raise ValueError(f"Pred/target shape mismatch: {tuple(outputs.shape)} vs {tuple(targets.shape)}")

        stats.update(y_true=targets, y_pred=outputs)

    return stats.finalize_score()
