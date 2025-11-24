from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


VIEW_SUFFIXES = ("a", "b")


def _load_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if list(data.files) == ["arr_0"] and data["arr_0"].dtype == object:
        return dict(data["arr_0"].item())
    return {k: data[k] for k in data.files}


def _load_any(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path)
    if suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            return loaded
        raise ValueError(f"Unsupported torch file structure in {path}")
    raise ValueError(f"Unsupported file extension for {path}")


def _find_key(data: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        if key in data:
            return key
    return None


def _standardize_view(data: Dict[str, Any], suffix: str) -> Dict[str, torch.Tensor]:
    point_key = _find_key(data, [f"P{suffix}", f"P_{suffix}", f"points_{suffix}", f"pointmap_{suffix}"])
    feat_key = _find_key(data, [f"F{suffix}", f"F_{suffix}", f"features_{suffix}"])
    conf_key = _find_key(data, [f"C{suffix}", f"C_{suffix}", f"confidence_{suffix}"])

    if point_key is None or feat_key is None or conf_key is None:
        raise KeyError(f"Could not find required keys for view '{suffix}' in input data")

    points = torch.as_tensor(data[point_key], dtype=torch.float32)
    features = torch.as_tensor(data[feat_key], dtype=torch.float32)
    confidence = torch.as_tensor(data[conf_key], dtype=torch.float32)

    if points.ndim == 3:
        h, w, _ = points.shape
        points = points.reshape(-1, 3)
        features = features.reshape(h * w, -1)
        confidence = confidence.reshape(h * w)
    elif points.ndim != 2:
        raise ValueError(f"Unexpected point dimensions for view '{suffix}': {points.shape}")

    return {
        "points": points,
        "features": features,
        "confidence": confidence,
    }


def _extract_pose(data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    pose = {}
    if "R_ab" in data:
        pose["R_ab"] = torch.as_tensor(data["R_ab"], dtype=torch.float32)
    if "t_ab" in data:
        pose["t_ab"] = torch.as_tensor(data["t_ab"], dtype=torch.float32)

    if "R_a" in data and "t_a" in data:
        pose["R_a"] = torch.as_tensor(data["R_a"], dtype=torch.float32)
        pose["t_a"] = torch.as_tensor(data["t_a"], dtype=torch.float32)
    if "R_b" in data and "t_b" in data:
        pose["R_b"] = torch.as_tensor(data["R_b"], dtype=torch.float32)
        pose["t_b"] = torch.as_tensor(data["t_b"], dtype=torch.float32)
    return pose


def _extract_matches(data: Dict[str, Any]) -> Optional[torch.Tensor]:
    match_key = _find_key(data, ["matches", "matches_ab", "correspondences"])
    if match_key is None:
        return None
    matches = torch.as_tensor(data[match_key])
    if matches.ndim != 2 or matches.shape[1] != 2:
        raise ValueError(f"Expected matches to have shape (N, 2), got {matches.shape}")
    return matches.long()


class PosePairDataset(Dataset):
    """Dataset that consumes MASt3R/DUNE inference outputs for pose solver training."""

    def __init__(self, pairs_list: str, root: Optional[str] = None):
        super().__init__()
        self.root = Path(root) if root is not None else None
        with open(pairs_list, "r", encoding="utf-8") as f:
            self.paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if not self.paths:
            raise ValueError(f"No training pairs found in {pairs_list}")

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or self.root is None:
            return candidate
        return self.root / candidate

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self._resolve_path(self.paths[idx])
        raw = _load_any(path)

        views = {suffix: _standardize_view(raw, suffix) for suffix in VIEW_SUFFIXES}
        pose = _extract_pose(raw)
        matches = _extract_matches(raw)

        if "R_ab" not in pose or "t_ab" not in pose:
            raise KeyError(f"Relative pose R_ab/t_ab missing in {path}")

        example: Dict[str, Any] = {
            "view_a": views["a"],
            "view_b": views["b"],
            "R_ab": pose["R_ab"],
            "t_ab": pose["t_ab"],
        }

        if matches is not None:
            example["matches"] = matches

        if "R_a" in pose and "t_a" in pose and "R_b" in pose and "t_b" in pose:
            example["R_a"] = pose["R_a"]
            example["t_a"] = pose["t_a"]
            example["R_b"] = pose["R_b"]
            example["t_b"] = pose["t_b"]
        return example


def pose_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keeps variable-resolution samples as a simple list for downstream processing."""

    return batch
