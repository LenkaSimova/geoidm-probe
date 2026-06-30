"""Dataset + metric + model-prediction diagnostics for the DROID-100 IDM probe.

Runs two diagnostics (World A "the metric hides the signal" vs World B "there is
no visible motion to learn"). See the root README for the full write-up.

    uv run diagnostics.py                              # data + baselines + proprio MLP
    uv run diagnostics.py --checkpoint idm_vision.pt   # also diagnose the Vision IDM
    uv run diagnostics.py --png-dir figures            # also write histogram PNGs
"""

import argparse
import glob
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_split(data_dir, split, with_frames=False):
    """Concatenate all episode .npz files for a split into flat arrays."""
    q_t, delta_qs, delta_gs, frames_t, frames_tk = [], [], [], [], []
    files = sorted(glob.glob(os.path.join(data_dir, split, "*.npz")))
    if not files:
        raise ValueError(f"No .npz files in {os.path.join(data_dir, split)}")
    for f in files:
        d = np.load(f)
        q_t.append(d["q_t"])
        delta_qs.append(d["delta_qs"])
        delta_gs.append(d["delta_gs"])
        if with_frames:
            frames_t.append(d["frames_t"])
            frames_tk.append(d["frames_tk"])
    out = {
        "q_t": np.concatenate(q_t).astype(np.float32),
        "delta_qs": np.concatenate(delta_qs).astype(np.float32),
        "delta_gs": np.concatenate(delta_gs).astype(np.float32),
    }
    if with_frames:
        out["frames_t"] = np.concatenate(frames_t)
        out["frames_tk"] = np.concatenate(frames_tk)
    return out


# --------------------------------------------------------------------------- #
# Diagnostic (2): magnitude / histogram of |Δq|
# --------------------------------------------------------------------------- #
def ascii_hist(values, bins, width=50, label="|dq|"):
    """Print a simple horizontal histogram of `values` over the given bin edges."""
    counts, edges = np.histogram(values, bins=bins)
    total = counts.sum()
    peak = counts.max() if counts.max() > 0 else 1
    print(f"  {label} histogram (N={total}):")
    for i in range(len(counts)):
        bar = "#" * int(width * counts[i] / peak)
        frac = counts[i] / total
        print(
            f"   [{edges[i]:7.4f},{edges[i + 1]:7.4f})  {bar:<{width}} {frac * 100:5.1f}%"
        )


def magnitude_diagnostics(dq, dg, joint_names=None):
    """Diagnostic 2: how big are the motions we ask the model to predict (World B)?"""
    J = dq.shape[1]
    joint_names = joint_names or [f"j{i}" for i in range(J)]
    absdq = np.abs(dq)

    print("\n=== DIAGNOSTIC 2: |Δq| MAGNITUDE / HISTOGRAM (k=5) ===")
    print(
        "How much motion is there to see? If the median per-joint |Δq| is "
        "~0.005 rad,\nthe arm barely moves between the two frames => World B "
        "(below visible threshold).\n"
    )

    print("Per-joint |Δq| (radians):")
    print(f"  {'joint':>6} {'median':>9} {'mean':>9} {'p90':>9} {'p99':>9} {'max':>9}")
    for j in range(J):
        col = absdq[:, j]
        print(
            f"  {joint_names[j]:>6} {np.median(col):9.4f} {col.mean():9.4f} "
            f"{np.quantile(col, 0.9):9.4f} {np.quantile(col, 0.99):9.4f} {col.max():9.4f}"
        )

    pooled = absdq.ravel()
    print(
        f"\nPooled |Δq| over all joints: median={np.median(pooled):.4f}  "
        f"mean={pooled.mean():.4f}  p90={np.quantile(pooled, 0.9):.4f}"
    )
    print(
        f"Gripper |Δg|:                median={np.median(np.abs(dg)):.4f}  "
        f"mean={np.abs(dg).mean():.4f}"
    )

    print("\nFraction of (sample, joint) entries below a motion threshold:")
    for thr in (0.002, 0.005, 0.01, 0.02, 0.05):
        print(f"  |Δq| < {thr:5.3f} rad : {(absdq < thr).mean() * 100:5.1f}%")

    print()
    # Log-ish bins from ~0 up to a high quantile, so the zero-spike is visible.
    hi = np.quantile(pooled, 0.995)
    bins = np.concatenate([[0.0], np.linspace(0.001, hi, 9)])
    ascii_hist(pooled, bins, label="pooled |Δq|")


# --------------------------------------------------------------------------- #
# Diagnostic (1): prediction quality (correlation / sign-agreement / R²)
# --------------------------------------------------------------------------- #
def per_joint_corr(pred, true):
    """Pearson r per joint. Returns nan for a constant predictor (std==0)."""
    J = true.shape[1]
    r = np.full(J, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(J):
            p, t = pred[:, j], true[:, j]
            if p.std() > 1e-9 and t.std() > 1e-9:
                r[j] = np.corrcoef(p, t)[0, 1]
    return r


def per_joint_sign_agreement(pred, true, motion_thr=0.02):
    """% sign agreement, restricted to samples with real motion (|true|>thr; chance 50%)."""
    J = true.shape[1]
    agree = np.full(J, np.nan)
    counts = np.zeros(J, dtype=int)
    for j in range(J):
        mask = np.abs(true[:, j]) > motion_thr
        counts[j] = mask.sum()
        if counts[j] > 0:
            agree[j] = (np.sign(pred[mask, j]) == np.sign(true[mask, j])).mean()
    return agree, counts


def per_joint_r2(pred, true):
    """R² per joint vs predicting the per-joint mean. <=0 means no better than mean."""
    J = true.shape[1]
    r2 = np.full(J, np.nan)
    for j in range(J):
        t = true[:, j]
        sst = ((t - t.mean()) ** 2).sum()
        sse = ((pred[:, j] - t) ** 2).sum()
        if sst > 1e-12:
            r2[j] = 1.0 - sse / sst
    return r2


def prediction_report(name, dq_pred, dq_true, dg_pred, dg_true, joint_names=None):
    """Diagnostic 1: is there recoverable signal in this predictor's outputs (World A)?"""
    J = dq_true.shape[1]
    joint_names = joint_names or [f"j{i}" for i in range(J)]

    mae = np.abs(dq_pred - dq_true).mean()
    g_mae = np.abs(dg_pred - dg_true).mean()
    r = per_joint_corr(dq_pred, dq_true)
    agree, counts = per_joint_sign_agreement(dq_pred, dq_true)
    r2 = per_joint_r2(dq_pred, dq_true)

    print(f"\n--- {name} ---")
    print(
        f"  joint MAE={mae:.4f}  gripper MAE={g_mae:.4f}  "
        f"pred |Δq| mean={np.abs(dq_pred).mean():.4f} (truth={np.abs(dq_true).mean():.4f})"
    )
    print(f"  {'joint':>6} {'corr':>8} {'sign%':>8} {'(nmot)':>8} {'R2':>8}")
    for j in range(J):
        rr = "   nan" if np.isnan(r[j]) else f"{r[j]:8.3f}"
        aa = "   nan" if np.isnan(agree[j]) else f"{agree[j] * 100:7.1f}"
        r2s = "   nan" if np.isnan(r2[j]) else f"{r2[j]:8.3f}"
        print(f"  {joint_names[j]:>6} {rr} {aa} {counts[j]:8d} {r2s}")
    mean_r = np.nan if np.all(np.isnan(r)) else np.nanmean(r)
    print(
        f"  mean per-joint corr = "
        f"{'nan (constant predictor)' if np.isnan(mean_r) else f'{mean_r:.3f}'}"
    )


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
def train_proprio_mlp(q_tr, dq_tr, dg_tr, q_te, epochs=15, device="cpu"):
    """Quick proprio-only MLP (q_t -> Δq,Δg). Reference: does q_t alone correlate?"""
    import torch
    import torch.nn as nn
    from train_proprio_baseline import BaselineProprioMLP

    net = BaselineProprioMLP().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.L1Loss()

    Xtr = torch.tensor(q_tr, device=device)
    dq = torch.tensor(dq_tr, device=device)
    dg = torch.tensor(dg_tr, device=device)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            dq_pred, dg_pred = net(Xtr[idx])
            loss = lossf(dq_pred, dq[idx]) + lossf(dg_pred, dg[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        dq_pred, dg_pred = net(torch.tensor(q_te, device=device))
    return dq_pred.cpu().numpy(), dg_pred.cpu().numpy()


def predict_vision_idm(checkpoint, data_dir, device="cpu"):
    """Load the trained VisionIDM checkpoint and run inference on the test split.

    Preprocessing mirrors train.py: uint8 HWC -> float CHW [0,1] -> ImageNet Normalize.
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    from model import VisionIDM

    ckpt = torch.load(checkpoint, map_location=device)
    use_proprio = ckpt.get("use_proprio", False) if isinstance(ckpt, dict) else False
    state = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )

    model = VisionIDM(use_proprio=use_proprio).to(device)
    model.load_state_dict(state)
    model.eval()

    te = load_split(data_dir, "test", with_frames=True)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    norm = T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)

    def to_batch(frames_uint8, idx):
        # uint8 (B,H,W,C) -> normalized float (B,C,H,W)
        x = torch.from_numpy(frames_uint8[idx]).permute(0, 3, 1, 2).float().div_(255.0)
        return norm(x).to(device)

    n = len(te["delta_qs"])
    q_t = torch.tensor(te["q_t"], device=device)
    preds_q, preds_g = [], []
    with torch.no_grad():
        for i in range(0, n, 128):
            idx = slice(i, i + 128)
            obs_t = to_batch(te["frames_t"], idx)
            obs_tk = to_batch(te["frames_tk"], idx)
            dq, dg = model(obs_t, obs_tk, q_t[idx] if use_proprio else None)
            preds_q.append(dq.cpu().numpy())
            preds_g.append(dg.cpu().numpy())
    return np.concatenate(preds_q), np.concatenate(preds_g)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./prepared_data")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained VisionIDM .pt (from train.py) to diagnose.",
    )
    ap.add_argument(
        "--no-proprio",
        action="store_true",
        help="Skip the inline proprio-MLP reference predictor.",
    )
    ap.add_argument(
        "--png-dir", default=None, help="If set, write histogram PNGs here."
    )
    args = ap.parse_args()
    np.set_printoptions(precision=4, suppress=True)

    JOINTS = [f"j{i}" for i in range(7)]

    tr = load_split(args.data_dir, "train")
    te = load_split(args.data_dir, "test", with_frames=False)
    dq_tr, dg_tr, q_tr = tr["delta_qs"], tr["delta_gs"], tr["q_t"]
    dq_te, dg_te, q_te = te["delta_qs"], te["delta_gs"], te["q_t"]

    print("=== SHAPES ===")
    print(f"train pairs: {len(dq_tr)} | test pairs: {len(dq_te)}")

    # ----- baseline MAE floors (context for everything below) -----
    print("\n=== BASELINE MAE FLOORS on TEST (any model must beat these) ===")
    print(
        f"ZERO   joint MAE: {np.abs(dq_te).mean():.4f} | gripper MAE: {np.abs(dg_te).mean():.4f}"
    )
    mq, mg = dq_tr.mean(0), dg_tr.mean(0)
    print(
        f"MEAN   joint MAE: {np.abs(dq_te - mq).mean():.4f} | gripper MAE: {np.abs(dg_te - mg).mean():.4f}"
    )

    # ----- Diagnostic 2: magnitude / histogram -----
    magnitude_diagnostics(dq_tr, dg_tr, JOINTS)

    # ----- Diagnostic 1: prediction quality -----
    print("\n=== DIAGNOSTIC 1: PREDICTION CORRELATION / SIGN-AGREEMENT / R² (test) ===")
    print(
        "corr>0 and sign% well above 50 => recoverable signal (World A: metric "
        "hides it).\nConstant predictors (zero/mean) have corr=nan and R²<=0 by "
        "construction — shown as the reference floor."
    )

    # Zero and mean baselines (constant predictors => corr nan, R2 <= 0).
    z_q = np.zeros_like(dq_te)
    z_g = np.zeros_like(dg_te)
    prediction_report("ZERO baseline", z_q, dq_te, z_g, dg_te, JOINTS)

    m_q = np.broadcast_to(mq, dq_te.shape)
    m_g = np.broadcast_to(mg, dg_te.shape)
    prediction_report("MEAN baseline", m_q, dq_te, m_g, dg_te, JOINTS)

    if not args.no_proprio:
        try:
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            pq, pg = train_proprio_mlp(q_tr, dq_tr, dg_tr, q_te, device=dev)
            prediction_report("PROPRIO MLP (q_t only)", pq, dq_te, pg, dg_te, JOINTS)
        except Exception as e:
            print(f"\n(proprio MLP skipped: {e})")

    if args.checkpoint:
        try:
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            vq, vg = predict_vision_idm(args.checkpoint, args.data_dir, device=dev)
            prediction_report("VISION IDM", vq, dq_te, vg, dg_te, JOINTS)
        except Exception as e:
            print(f"\n(vision IDM skipped: {e})")
    else:
        print(
            "\n(no --checkpoint given; train.py now saves idm_vision.pt — pass it "
            "to diagnose the vision model.)"
        )


if __name__ == "__main__":
    main()
