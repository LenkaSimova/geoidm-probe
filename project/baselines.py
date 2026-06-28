import torch
import torch.nn as nn


class BaselineZeroMotion:
    """Predicts absolutely no movement for joints and gripper."""

    def predict(self, batch_size, device="cpu"):
        # 7 DoF for joints, 1 for gripper
        pred_q = torch.zeros(batch_size, 7, device=device)
        pred_g = torch.zeros(batch_size, 1, device=device)
        return pred_q, pred_g


class BaselineMeanDelta:
    """Predicts the average movement observed in the training dataset."""

    def __init__(self, train_delta_q: torch.Tensor, train_delta_g: torch.Tensor):
        # Calculate means during initialization based on training data
        self.mean_q = train_delta_q.mean(dim=0)
        self.mean_g = train_delta_g.mean(dim=0)

    def predict(self, batch_size, device="cpu"):
        # Expand the mean to match the batch size for evaluation
        pred_q = self.mean_q.to(device).expand(batch_size, -1)
        pred_g = self.mean_g.to(device).expand(batch_size, -1)
        return pred_q, pred_g


class BaselineProprioMLP(nn.Module):
    """
    Predicts movement using ONLY the current joint position (q_t).
    """

    def __init__(self):
        super().__init__()
        # Input: q_t (7 dimensions)
        # Output: Δq (7 dimensions) + Δg (1 dimension)
        self.net = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)
        )

    def forward(self, q_t):
        out = self.net(q_t)
        # Split output into joint delta (first 7) and gripper delta (last 1)
        return out[:, :7], out[:, 7:]
