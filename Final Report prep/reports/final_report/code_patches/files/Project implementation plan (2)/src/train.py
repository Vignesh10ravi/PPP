"""Sobolev training of the pICNN against Neo-Hookean energy and stress.

Usage:
    python -m src.train --epochs 1500 --ckpt checkpoints/phase1.pt
"""
import argparse
import os

import torch

from .constitutive import stress
from .neohooke import make_dataset
from .picnn import HyperelasticNN


def train(
    epochs=1500,
    batch=512,
    lr=2e-3,
    lam_psi=1.0,
    lam_P=10.0,
    n_data=20000,
    device="cpu",
    seed=0,
    ckpt=None,
    resume=None,
    verbose=True,
):
    torch.manual_seed(seed)

    F_tr, psi_tr, P_tr = make_dataset(
        n_data,
        device=device,
        seed=seed,
    )
    F_tr = F_tr.double()
    psi_tr = psi_tr.double()
    P_tr = P_tr.double()

    model = HyperelasticNN().to(device).double()
    history = []
    optimizer_state = None

    if resume:
        resumed = torch.load(
            resume,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(resumed["state_dict"])
        history = list(resumed.get("history", []))
        optimizer_state = resumed.get("optimizer_state_dict")
        if verbose:
            print(f"resumed model from {resume}")
            print(f"existing history: {len(history)} epochs")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

        # Honor the command-line learning rate after restoring Adam state.
        for group in optimizer.param_groups:
            group["lr"] = lr

    start_epoch = len(history)

    for step in range(epochs):
        indices = torch.randint(
            0,
            len(F_tr),
            (batch,),
            device=device,
        )
        F = F_tr[indices].clone()

        P_hat = stress(model, F)
        psi_hat = model.energy(F)

        loss = (
            lam_psi * ((psi_hat - psi_tr[indices]) ** 2).mean()
            + lam_P * ((P_hat - P_tr[indices]) ** 2).mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if verbose and step % 200 == 0:
            epoch = start_epoch + step
            print(f"epoch {epoch:5d}  loss {loss.item():.3e}")

    if ckpt:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            ckpt,
        )
        if verbose:
            print(f"saved -> {ckpt}")

    return model, history


def load_model(ckpt, device="cpu"):
    checkpoint = torch.load(
        ckpt,
        map_location=device,
        weights_only=True,
    )
    state = dict(checkpoint["state_dict"])
    legacy = "feature_mean" not in state
    model = HyperelasticNN(
        reference_correction="affine" if legacy else "pressure"
    ).to(device).double()
    # Phase 1 checkpoints predate the input-normalization buffers added for
    # Phase 2 and the later objective reference correction. They were trained
    # on raw invariant features (identity normalization) with the affine
    # reference correction, so both are restored for legacy checkpoints.
    for name, buffer in model.state_dict().items():
        if name.endswith(("_mean", "_scale")) and name not in state:
            state[name] = (torch.zeros_like(buffer) if name.endswith("_mean")
                           else torch.ones_like(buffer))
    model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--ckpt", default="checkpoints/phase1.pt")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    train(
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        ckpt=args.ckpt,
        resume=args.resume,
    )
