#!/usr/bin/env python3
"""Training entrypoint for the transformer-based pose solver."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mast3r.pose_solver import PosePairDataset, PoseSolver, PoseSolverLoss, pose_collate
from mast3r.pose_solver.utils import batch_to_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a pose solver on MASt3R/DUNE inference outputs")
    parser.add_argument("--pairs-list", type=str, required=True, help="Text file containing paths to pair npz/pt files")
    parser.add_argument("--val-pairs-list", type=str, default=None, help="Optional validation set")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory to prepend to relative paths in pairs lists")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=1024, help="Number of tokens per view")
    parser.add_argument("--selection-method", type=str, choices=["topk", "fps"], default="topk")
    parser.add_argument("--lambda-t", type=float, default=1.0, help="Weight for translation angle loss")
    parser.add_argument("--lambda-pp", type=float, default=0.0, help="Weight for point-to-point consistency")
    parser.add_argument("--feature-dim", type=int, default=None, help="Input feature dimension; defaults to lazy init")
    return parser


def run_epoch(
    model: PoseSolver,
    criterion: PoseSolverLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
) -> dict:
    epoch_stats = {"loss": 0.0, "rot_loss": 0.0, "trans_loss": 0.0, "pp_loss": 0.0, "steps": 0}
    if train:
        model.train()
    else:
        model.eval()

    for batch in dataloader:
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            losses = criterion(outputs, batch)
            loss = losses["loss"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_stats["loss"] += loss.item()
        epoch_stats["rot_loss"] += losses["rot_loss"].item()
        epoch_stats["trans_loss"] += losses["trans_loss"].item()
        if losses["pp_loss"] is not None:
            epoch_stats["pp_loss"] += losses["pp_loss"].item()
        epoch_stats["steps"] += 1

    for key in ["loss", "rot_loss", "trans_loss", "pp_loss"]:
        epoch_stats[key] = epoch_stats[key] / max(epoch_stats["steps"], 1)
    return epoch_stats


def main(args: argparse.Namespace):
    device = torch.device(args.device)
    train_dataset = PosePairDataset(args.pairs_list, root=args.data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pose_collate,
    )

    val_loader = None
    if args.val_pairs_list:
        val_dataset = PosePairDataset(args.val_pairs_list, root=args.data_root)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pose_collate,
        )

    model = PoseSolver(k=args.k, selection_method=args.selection_method, feature_dim=args.feature_dim)
    model.to(device)
    criterion = PoseSolverLoss(lambda_t=args.lambda_t, lambda_pp=args.lambda_pp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train_stats = run_epoch(model, criterion, train_loader, optimizer, device, train=True)
        msg = f"Epoch {epoch+1}/{args.epochs} - train loss: {train_stats['loss']:.4f}, rot: {train_stats['rot_loss']:.4f}, trans: {train_stats['trans_loss']:.4f}"
        if args.lambda_pp > 0:
            msg += f", pp: {train_stats['pp_loss']:.4f}"
        if val_loader is not None:
            val_stats = run_epoch(model, criterion, val_loader, optimizer, device, train=False)
            msg += f" | val loss: {val_stats['loss']:.4f}"
        print(msg)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    ckpt_path = checkpoint_dir / "pose_solver_latest.pth"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
