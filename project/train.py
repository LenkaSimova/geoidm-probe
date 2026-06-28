import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T

# Import the model you built in the previous step
from model import VisionIDM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "./prepared_data"
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-4  # Slightly lower for fine-tuning ResNet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_PROPRIO = False  # Set to True for Stretch S3 (Vision + Proprio ablation)


# --- 1. Vision Dataset with In-Memory Loading ---
class DroidVisionDataset(Dataset):
    """
    Loads images and kinematics from the .npz files.
    Since DROID-100 is small (~2GB), we load it entirely into RAM for speed.
    """

    def __init__(self, split_dir, transform=None):
        super().__init__()
        file_pattern = os.path.join(DATA_DIR, split_dir, "*.npz")
        self.files = sorted(glob.glob(file_pattern))
        self.transform = transform

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
        self.frames_t = np.concatenate(f_t, axis=0)
        self.frames_tk = np.concatenate(f_tk, axis=0)
        self.q_t = torch.tensor(np.concatenate(q_t, axis=0), dtype=torch.float32)
        self.delta_qs = torch.tensor(np.concatenate(dq, axis=0), dtype=torch.float32)
        self.delta_gs = torch.tensor(np.concatenate(dg, axis=0), dtype=torch.float32)

    def __len__(self):
        return len(self.delta_qs)

    def __getitem__(self, idx):
        # The frames are stored as uint8 arrays: shape (H, W, C)
        img_t = self.frames_t[idx]
        img_tk = self.frames_tk[idx]

        if self.transform:
            img_t = self.transform(img_t)
            img_tk = self.transform(img_tk)

        return img_t, img_tk, self.q_t[idx], self.delta_qs[idx], self.delta_gs[idx]


# --- 2. Main Training Loop ---
def main():
    print(f"Using device: {DEVICE}")
    print(f"Proprioception Ablation (Stretch S3): {'ON' if USE_PROPRIO else 'OFF'}")

    # Standard ImageNet normalization because we use pre-trained ResNet weights
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )

    # 1. Load Data
    train_dataset = DroidVisionDataset("train", transform=transform)
    test_dataset = DroidVisionDataset(
        "test", transform=transform
    )  # Held-out episode split

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
            obs_t = obs_t.to(DEVICE)
            obs_tk = obs_tk.to(DEVICE)
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
            obs_t = obs_t.to(DEVICE)
            obs_tk = obs_tk.to(DEVICE)
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
