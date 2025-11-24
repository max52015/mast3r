from typing import Tuple

import torch


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Converts a 6D representation to a valid rotation matrix using Gram-Schmidt."""

    if x.shape[-1] != 6:
        raise ValueError(f"Expected last dim to be 6, got {x.shape}")
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def geodesic_distance(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Returns the geodesic distance (in radians) between two rotation matrices."""

    R_rel = torch.matmul(R_pred, R_gt.transpose(-1, -2))
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps)
    return torch.arccos(cosine)


def translation_direction_loss(t_pred: torch.Tensor, t_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Angle between predicted and ground-truth translation direction."""

    t_pred = torch.nn.functional.normalize(t_pred, dim=-1, eps=eps)
    t_gt = torch.nn.functional.normalize(t_gt, dim=-1, eps=eps)
    cosine = (t_pred * t_gt).sum(-1).clamp(-1 + eps, 1 - eps)
    return torch.arccos(cosine)


def batch_to_device(batch: list, device: torch.device) -> list:
    moved = []
    for sample in batch:
        moved.append({k: _move_value(v, device) for k, v in sample.items()})
    return moved


def _move_value(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_value(v, device) for k, v in value.items()}
    return value


def log_translation_stats(loss: torch.Tensor) -> Tuple[float, float]:
    return float(loss.mean().item()), float(loss.max().item())
