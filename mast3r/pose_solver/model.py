from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .utils import rotation_6d_to_matrix


@dataclass
class TokenSelection:
    points: torch.Tensor
    features: torch.Tensor
    confidence: torch.Tensor
    indices: torch.Tensor


class TokenSelector:
    def __init__(self, k: int = 1024, method: str = "topk", use_confidence_bias: bool = True):
        self.k = k
        self.method = method
        self.use_confidence_bias = use_confidence_bias

    def _topk(self, confidence: torch.Tensor, k: int) -> torch.Tensor:
        if confidence.numel() < k:
            return torch.arange(confidence.numel(), device=confidence.device)
        return torch.topk(confidence, k=k).indices

    def _fps(self, points: torch.Tensor, confidence: torch.Tensor, k: int) -> torch.Tensor:
        # simple torch FPS implementation; operates on CPU/GPU tensors.
        N = points.shape[0]
        k = min(k, N)
        if k == N:
            return torch.arange(N, device=points.device)
        selected = torch.zeros(k, dtype=torch.long, device=points.device)
        weights = confidence if self.use_confidence_bias else torch.ones_like(confidence)
        # start from highest confidence point
        selected[0] = torch.argmax(weights)
        dist = torch.full((N,), float("inf"), device=points.device)
        for i in range(1, k):
            last = points[selected[i - 1]].unsqueeze(0)
            dist = torch.minimum(dist, torch.norm(points - last, dim=1))
            scores = dist * weights
            selected[i] = torch.argmax(scores)
        return selected

    def select(self, view: Dict[str, torch.Tensor], paired_length: int) -> TokenSelection:
        points, features, confidence = view["points"], view["features"], view["confidence"].flatten()
        k = min(self.k, points.shape[0], paired_length)
        if self.method == "fps":
            idx = self._fps(points, confidence, k)
        else:
            idx = self._topk(confidence, k)
        return TokenSelection(points[idx], features[idx], confidence[idx], idx)


class TokenEmbedding(nn.Module):
    def __init__(self, d_model: int = 256, d_p: int = 64, d_f: int = 128, d_c: int = 32, feat_dim: Optional[int] = None):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, d_p),
            nn.GELU(),
            nn.Linear(d_p, d_p),
        )
        feat_proj = nn.Linear(feat_dim, d_f) if feat_dim is not None else nn.LazyLinear(d_f)
        self.feat_mlp = nn.Sequential(
            feat_proj,
            nn.GELU(),
            nn.Linear(d_f, d_f),
        )
        self.conf_mlp = nn.Sequential(
            nn.Linear(1, d_c),
            nn.GELU(),
            nn.Linear(d_c, d_c),
        )
        self.view_embedding = nn.Parameter(torch.randn(2, d_model))
        self.proj = nn.Linear(d_p + d_f + d_c, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, selection: TokenSelection, view_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p_embed = self.point_mlp(selection.points)
        f_embed = self.feat_mlp(selection.features)
        c_embed = self.conf_mlp(selection.confidence.unsqueeze(-1))
        x = torch.cat([p_embed, f_embed, c_embed], dim=-1)
        x = self.proj(x)
        gate = torch.sigmoid(self.alpha * selection.confidence.unsqueeze(-1) + self.beta)
        x = self.norm(x * gate + self.view_embedding[view_idx])
        return x, selection.confidence


class GeometryEncoder(nn.Module):
    def __init__(self, d_g: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, d_g),
            nn.GELU(),
            nn.Linear(d_g, d_g),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, N, 3)
        rel = positions[:, :, None, :] - positions[:, None, :, :]
        dist = torch.norm(rel, dim=-1, keepdim=True)
        dir_norm = rel / (dist + 1e-6)
        geom = torch.cat([dist, dir_norm], dim=-1)
        return self.mlp(geom)


class GeometryAwareAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, d_g: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.d_head = dim // heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.k_geom = nn.Linear(d_g, dim)
        self.geom_bias = nn.Linear(d_g, heads, bias=False)
        self.conf_gamma = nn.Parameter(torch.tensor(1.0))
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, geom: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.heads, self.d_head).permute(0, 2, 1, 3)
        k_base = self.k_proj(x).view(B, N, self.heads, self.d_head)
        v = self.v_proj(x).view(B, N, self.heads, self.d_head)

        geom_k = self.k_geom(geom).view(B, N, N, self.heads, self.d_head)
        k = k_base.unsqueeze(1) + geom_k

        attn_logits = (q.permute(0, 2, 1, 3).unsqueeze(2) * k).sum(-1).permute(0, 3, 1, 2)
        geom_bias = self.geom_bias(geom).permute(0, 3, 1, 2)
        attn_logits = attn_logits * self.scale + geom_bias

        conf_bias = torch.log((confidence + 1e-6).unsqueeze(1) * (confidence + 1e-6).unsqueeze(2))
        attn_logits = attn_logits + self.conf_gamma * conf_bias.unsqueeze(1)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(B, N, self.dim)
        return self.out(out)


class PoseSolverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, d_g: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GeometryAwareAttention(dim, heads=heads, d_g=d_g, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, geom: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), geom, confidence)
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn = torch.einsum("bd,d->b", tokens, self.query)
        weights = torch.softmax(attn, dim=0)
        return (tokens * weights.unsqueeze(-1)).sum(dim=0)


class RelativePoseHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.rot_out = nn.Linear(dim, 6)
        self.trans_out = nn.Linear(dim, 3)

    def forward(self, g_a: torch.Tensor, g_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([g_a, g_b, g_a - g_b, g_a * g_b], dim=-1)
        fused = self.mlp(h)
        rot_6d = self.rot_out(fused)
        rot = rotation_6d_to_matrix(rot_6d)
        trans = torch.nn.functional.normalize(self.trans_out(fused), dim=-1)
        return rot, trans


class PoseSolver(nn.Module):
    def __init__(
        self,
        k: int = 1024,
        d_model: int = 256,
        d_g: int = 64,
        depth: int = 4,
        heads: int = 4,
        token_embed_dim: Tuple[int, int, int] = (64, 128, 32),
        selection_method: str = "topk",
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        self.selector = TokenSelector(k=k, method=selection_method)
        self.embedder = TokenEmbedding(
            d_model=d_model,
            d_p=token_embed_dim[0],
            d_f=token_embed_dim[1],
            d_c=token_embed_dim[2],
            feat_dim=feature_dim,
        )
        self.geom_encoder = GeometryEncoder(d_g=d_g)
        self.blocks = nn.ModuleList(
            [PoseSolverBlock(dim=d_model, heads=heads, d_g=d_g) for _ in range(depth)]
        )
        self.pool = AttentionPooling(dim=d_model)
        self.pose_head = RelativePoseHead(dim=d_model)

    def _prepare_tokens(self, view: Dict[str, torch.Tensor], paired_length: int, view_idx: int) -> Tuple[torch.Tensor, torch.Tensor, TokenSelection]:
        selection = self.selector.select(view, paired_length)
        tokens, confidence = self.embedder(selection, view_idx)
        return tokens, confidence, selection

    def _forward_single(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        paired_k = min(self.selector.k, batch["view_a"]["points"].shape[0], batch["view_b"]["points"].shape[0])
        tokens_a, conf_a, sel_a = self._prepare_tokens(batch["view_a"], paired_k, view_idx=0)
        tokens_b, conf_b, sel_b = self._prepare_tokens(batch["view_b"], paired_k, view_idx=1)

        tokens = torch.cat([tokens_a, tokens_b], dim=0).unsqueeze(0)
        positions = torch.cat([sel_a.points, sel_b.points], dim=0).unsqueeze(0)
        confidence = torch.cat([conf_a, conf_b], dim=0).unsqueeze(0)

        geom = self.geom_encoder(positions)
        x = tokens
        for block in self.blocks:
            x = block(x, geom, confidence)

        x = x.squeeze(0)
        tokens_a_out, tokens_b_out = x[: tokens_a.shape[0]], x[tokens_a.shape[0] :]
        g_a = self.pool(tokens_a_out)
        g_b = self.pool(tokens_b_out)
        R_ab, t_ab = self.pose_head(g_a, g_b)
        return {
            "R_ab": R_ab,
            "t_ab": t_ab,
            "tokens_a": tokens_a_out,
            "tokens_b": tokens_b_out,
            "sel_a": sel_a,
            "sel_b": sel_b,
        }

    def forward(self, batch: Dict[str, torch.Tensor] | List[Dict[str, torch.Tensor]]):
        if isinstance(batch, list):
            return [self._forward_single(sample) for sample in batch]
        return self._forward_single(batch)
