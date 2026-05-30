"""Loss functions for training."""

import torch
import torch.nn.functional as F


def wmse_loss(
    pred:    torch.Tensor,   # (batch, n_genes)
    target:  torch.Tensor,   # (batch, n_genes)
    weights: torch.Tensor,   # (n_genes,)
) -> torch.Tensor:
    """Weighted MSE — matches the contest Mejia scoring metric."""
    return ((pred - target) ** 2 * weights.unsqueeze(0)).mean()


def kl_variance_loss(
    pred_var: torch.Tensor,   # (batch, n_genes)
    emp_var:  torch.Tensor,   # (n_genes,) empirical gene variance from training set
) -> torch.Tensor:
    """Log-space MSE between predicted and empirical variance (KL proxy)."""
    target = emp_var.unsqueeze(0).expand_as(pred_var)
    return F.mse_loss(torch.log(pred_var + 1e-8), torch.log(target + 1e-8))
