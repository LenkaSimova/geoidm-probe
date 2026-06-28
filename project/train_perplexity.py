import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Import the model you built in the previous step
from model import VisionIDM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "./prepared_data"
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-4  # Slightly lower for fine-tuning ResNet
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
USE_PROPRIO = False  # Set to True for Stretch S3 (Vision + Proprio ablation)


class DroidVisionDataset(Dataset):
    """
    Loads images and kinematics from the .npz files.
    Since DROID-100 is small (~2GB), we load it entirely into RAM for speed.
    Images are stored as uint8 CHW tensors to avoid a 4× memory blowup from float32.
    """

    def __init__(self, split_dir):
        super().__init__()
        file_pattern = os.path.join(DATA_DIR, split_dir, "*.npz")
        self.files = sorted(glob.glob(file_pattern))

        if not self.files:
            raise ValueError(f"No .npz files found in {split_dir}. Check data prep.")

        # Lists to hold arrays before concatenating
        f_t, f_tk, q_t, dq, dg = [], [], [], [], []

        print(f"Loading {split_dir} data into memory...")
        for f in tqdm(self.files, desc="Reading .npz files", leave=False):
            data = np.load(f)
            f_t.append(data["frames_t"])
            f_tk.append(data["frames_tk"])
            q_t.append(data["q_t"])
            dq.append(data["delta_qs"])
            dg.append(data["delta_gs"])

        # Concatenate all episodes into flat arrays
        self.q_t = torch.tensor(np.concatenate(q_t, axis=0), dtype=torch.float32)
        self.delta_qs = torch.tensor(np.concatenate(dq, axis=0), dtype=torch.float32)
        self.delta_gs = torch.tensor(np.concatenate(dg, axis=0), dtype=torch.float32)

        # Store images as uint8 CHW tensors (same memory as raw numpy).
        # The HWC -> CHW permutation is done once here; the float conversion
        # and ImageNet normalization happen per-batch in the training loop.
        print(f"Converting images to CHW for {split_dir}...")
        self.frames_t = (
            torch.from_numpy(np.concatenate(f_t, axis=0))
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # (N, C, H, W) uint8
        del f_t

        self.frames_tk = (
            torch.from_numpy(np.concatenate(f_tk, axis=0))
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # (N, C, H, W) uint8
        del f_tk

    def __len__(self):
        return len(self.delta_qs)

    def __getitem__(self, idx):
        return (
            self.frames_t[idx],
            self.frames_tk[idx],
            self.q_t[idx],
            self.delta_qs[idx],
            self.delta_gs[idx],
        )


def normalize_batch(img_batch, mean, std):
    """Convert a uint8 CHW batch to float32 and apply ImageNet normalization."""
    return img_batch.float().div_(255.0).sub_(mean).div_(std)


# --- 2. Main Training Loop ---
def main():
    print(f"Using device: {DEVICE}")
    print(f"Proprioception Ablation (Stretch S3): {'ON' if USE_PROPRIO else 'OFF'}")

    # Standard ImageNet normalization (applied per-batch in the training loop)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    imagenet_mean = (
        torch.tensor(list(weights.transforms().mean)).view(1, 3, 1, 1).to(DEVICE)
    )
    imagenet_std = (
        torch.tensor(list(weights.transforms().std)).view(1, 3, 1, 1).to(DEVICE)
    )

    # 1. Load Data
    train_dataset = DroidVisionDataset("train")
    test_dataset = DroidVisionDataset("test")  # Held-out episode split

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 2. Initialize Model, Loss, Optimizer
    model = VisionIDM(use_proprio=USE_PROPRIO).to(DEVICE)
    criterion_q = nn.L1Loss()  # Using L1 because metric is MAE
    criterion_g = nn.L1Loss()

    # --- PHASE 1 SETUP: LINEAR PROBING ---
    print("\n[Phase 1] Freezing ResNet backbone. Training action head only...")
    model.freeze_entire_encoder()

    # Optimizer only knows about the action head
    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": 0.0},  # Frozen, LR = 0
            {"params": model.action_head.parameters(), "lr": LEARNING_RATE},
        ]
    )

    # 3. Training
    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        # --- PHASE 2 SETUP: PARTIAL FINE-TUNING ---
        if epoch == 6:
            print("\n[Phase 2] Unfreezing top ResNet layers. Dropping learning rate...")
            model.unfreeze_top_encoder_layers()

            # Update the learning rate of the encoder group (index 0)
            optimizer.param_groups[0]["lr"] = 1e-5

        model.train()
        train_loss = 0.0

        for obs_t, obs_tk, q_t, dq_true, dg_true in tqdm(
            train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False
        ):
            obs_t = normalize_batch(obs_t.to(DEVICE), imagenet_mean, imagenet_std)
            obs_tk = normalize_batch(obs_tk.to(DEVICE), imagenet_mean, imagenet_std)
            q_t = q_t.to(DEVICE)
            dq_true = dq_true.to(DEVICE)
            dg_true = dg_true.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            dq_pred, dg_pred = model(obs_t, obs_tk, q_t)

            # Sum losses
            loss_q = criterion_q(dq_pred, dq_true)
            loss_g = criterion_g(dg_pred, dg_true)
            loss = loss_q + loss_g

            # Backprop
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs_t.size(0)

        train_loss /= len(train_dataset)
        print(f"Epoch {epoch:02d} | Train Loss (MAE sum): {train_loss:.5f}")

    # 4. Evaluation (The Prerequisite Gate)
    print("\n--- Evaluating IDM on Held-out Episodes ---")
    model.eval()
    total_q_mae = 0.0
    total_g_mae = 0.0

    with torch.no_grad():
        for obs_t, obs_tk, q_t, dq_true, dg_true in tqdm(
            test_loader, desc="Evaluating"
        ):
            obs_t = normalize_batch(obs_t.to(DEVICE), imagenet_mean, imagenet_std)
            obs_tk = normalize_batch(obs_tk.to(DEVICE), imagenet_mean, imagenet_std)
            q_t = q_t.to(DEVICE)
            dq_true = dq_true.to(DEVICE)
            dg_true = dg_true.to(DEVICE)

            dq_pred, dg_pred = model(obs_t, obs_tk, q_t)

            batch_q_mae = torch.abs(dq_pred - dq_true).mean().item()
            batch_g_mae = torch.abs(dg_pred - dg_true).mean().item()

            total_q_mae += batch_q_mae * obs_t.size(0)
            total_g_mae += batch_g_mae * obs_t.size(0)

    final_q_mae = total_q_mae / len(test_dataset)
    final_g_mae = total_g_mae / len(test_dataset)

    print("\n[Vision IDM Results]")
    print(f"Joint MAE:   {final_q_mae:.5f} radians")
    print(f"Gripper MAE: {final_g_mae:.5f}")
    print("\n>>> Compare these numbers to your 3 baselines! <<<")
    print(
        "If these are lower than the proprio-only (q_t) baseline, your model successfully learns from vision."
    )


if __name__ == "__main__":
    main()
