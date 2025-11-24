from typing import Dict, List, Optional

import torch

from .utils import geodesic_distance, translation_direction_loss


def _point_to_point_loss(
    pred: Dict[str, torch.Tensor],
    sample: Dict[str, torch.Tensor],
    weight_matches_by_conf: bool = True,
) -> Optional[torch.Tensor]:
    matches = sample.get("matches")
    if matches is None or matches.numel() == 0:
        return None

    points_a = sample["view_a"]["points"]
    points_b = sample["view_b"]["points"]
    conf_a = sample["view_a"].get("confidence")
    conf_b = sample["view_b"].get("confidence")

    idx_a = matches[:, 0].long()
    idx_b = matches[:, 1].long()

    if idx_a.max() >= points_a.shape[0] or idx_b.max() >= points_b.shape[0]:
        raise ValueError("Match indices exceed available points")

    pa = points_a[idx_a]
    pb = points_b[idx_b]

    pb_pred = torch.matmul(pred["R_ab"], pa.transpose(0, 1)).transpose(0, 1) + pred["t_ab"].unsqueeze(0)
    l1 = (pb_pred - pb).abs().sum(dim=-1)

    if weight_matches_by_conf and conf_a is not None and conf_b is not None:
        weights = conf_a[idx_a] * conf_b[idx_b]
        l1 = l1 * weights
    return l1.mean()


class PoseSolverLoss:
    def __init__(self, lambda_t: float = 1.0, lambda_pp: float = 0.0):
        self.lambda_t = lambda_t
        self.lambda_pp = lambda_pp

    def __call__(self, preds: Dict[str, torch.Tensor] | List[Dict[str, torch.Tensor]], batch: Dict[str, torch.Tensor] | List[Dict[str, torch.Tensor]]):
        if isinstance(preds, dict):
            preds = [preds]
        if isinstance(batch, dict):
            batch = [batch]

        rot_losses = []
        trans_losses = []
        pp_losses = []
        for pred, sample in zip(preds, batch):
            rot_losses.append(geodesic_distance(pred["R_ab"], sample["R_ab"]))
            trans_losses.append(translation_direction_loss(pred["t_ab"], sample["t_ab"]))
            if self.lambda_pp > 0:
                pp = _point_to_point_loss(pred, sample)
                if pp is not None:
                    pp_losses.append(pp)

        rot_loss = torch.stack(rot_losses).mean()
        trans_loss = torch.stack(trans_losses).mean() if trans_losses else torch.tensor(0.0, device=rot_loss.device)
        total = rot_loss + self.lambda_t * trans_loss

        pp_loss = None
        if pp_losses:
            pp_loss = torch.stack(pp_losses).mean()
            total = total + self.lambda_pp * pp_loss

        return {
            "loss": total,
            "rot_loss": rot_loss,
            "trans_loss": trans_loss,
            "pp_loss": pp_loss,
        }
