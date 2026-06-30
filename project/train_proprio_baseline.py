import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "./prepared_data"
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. Dataset specifically for Proprioception ---
class ProprioDataset(Dataset):
    """
    Loads only the proprioceptive state (q_t) and targets (Δq, Δg).
    Ignores images entirely to keep RAM usage trivial.
    """

    def __init__(self, split_dir):
        super().__init__()
        # Load all .npz files for the specified split (train or test)
        file_pattern = os.path.join(DATA_DIR, split_dir, "*.npz")
        self.files = sorted(glob.glob(file_pattern))

        if not self.files:
            raise ValueError(
                f"No .npz files found in {split_dir}. Run data prep first."
            )

        # Read everything into memory (feasible because we drop images)
        q_t_list, dq_list, dg_list = [], [], []

        for f in self.files:
            data = np.load(f)
            q_t_list.append(torch.tensor(data["q_t"], dtype=torch.float32))
            dq_list.append(torch.tensor(data["delta_qs"], dtype=torch.float32))
            dg_list.append(torch.tensor(data["delta_gs"], dtype=torch.float32))

        self.q_t = torch.cat(q_t_list, dim=0)
        self.delta_qs = torch.cat(dq_list, dim=0)
        self.delta_gs = torch.cat(dg_list, dim=0)

    def __len__(self):
        return len(self.q_t)

    def __getitem__(self, idx):
        return self.q_t[idx], self.delta_qs[idx], self.delta_gs[idx]


# --- 2. The Model ---
class BaselineProprioMLP(nn.Module):
    """Predicts Δq and Δg using ONLY the current joint position (q_t)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 7 for joints, 1 for gripper
        )

    def forward(self, q_t):
        out = self.net(q_t)
        return out[:, :7], out[:, 7:]


# --- 3. Training and Evaluation Loop ---
def main():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = ProprioDataset("train")
    test_dataset = ProprioDataset("test")  # This enforces the held-out-episode split

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model, Loss, Optimizer
    model = BaselineProprioMLP().to(DEVICE)
    # Using L1 Loss because our target metric is Mean Absolute Error (MAE)
    criterion_q = nn.L1Loss()
    criterion_g = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for q_t, dq_true, dg_true in tqdm(
            train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False
        ):
            q_t = q_t.to(DEVICE)
            dq_true = dq_true.to(DEVICE)
            dg_true = dg_true.to(DEVICE)

            optimizer.zero_grad()
            dq_pred, dg_pred = model(q_t)

            # Combine joint and gripper loss (can weight differently if needed)
            loss_q = criterion_q(dq_pred, dq_true)
            loss_g = criterion_g(dg_pred, dg_true)
            loss = loss_q + loss_g

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * q_t.size(0)

        train_loss /= len(train_dataset)
        print(f"Epoch {epoch:02d} | Train Loss (MAE): {train_loss:.5f}")

    # 4. Evaluation Loop (The Prerequisite Gate)
    print("\n--- Evaluating Prerequisite Gate (Held-out Episodes) ---")
    model.eval()
    total_q_mae = 0.0
    total_g_mae = 0.0

    with torch.no_grad():
        for q_t, dq_true, dg_true in test_loader:
            q_t = q_t.to(DEVICE)
            dq_true = dq_true.to(DEVICE)
            dg_true = dg_true.to(DEVICE)

            dq_pred, dg_pred = model(q_t)

            # Compute batch MAE
            batch_q_mae = torch.abs(dq_pred - dq_true).mean().item()
            batch_g_mae = torch.abs(dg_pred - dg_true).mean().item()

            # Accumulate
            total_q_mae += batch_q_mae * q_t.size(0)
            total_g_mae += batch_g_mae * q_t.size(0)

    final_q_mae = total_q_mae / len(test_dataset)
    final_g_mae = total_g_mae / len(test_dataset)

    print(f"Proprio-Only Joint MAE:   {final_q_mae:.5f} radians")
    print(f"Proprio-Only Gripper MAE: {final_g_mae:.5f}")


if __name__ == "__main__":
    main()
