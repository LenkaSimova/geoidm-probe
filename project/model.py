import torch
import torch.nn as nn
import torchvision.models as models


class VisionIDM(nn.Module):
    """
    Two-frame Vision Inverse Dynamics Model.
    Inputs: obs_t (RGB), obs_tk (RGB), optional q_t (proprioception).
    Outputs: Δq (7-DoF joint change), Δg (1-DoF gripper change).
    """

    def __init__(self, use_proprio=False):
        super().__init__()
        self.use_proprio = use_proprio

        # 1. Shared Vision Encoder (ResNet18, ImageNet weights)
        # Using ResNet18 keeps it ~12M parameters and easily trainable on a consumer GPU.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Strip the final classification layer (fc) to extract the raw 512-dim features.
        # The output of the layer before 'fc' is exactly what we need after pooling.
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet18 outputs 512 features per image. Two images = 1024 features.
        cnn_out_dim = 512
        mlp_input_dim = cnn_out_dim * 2

        # Optional: Add 7 dimensions if we are concatenating the current joint state (q_t)
        if self.use_proprio:
            mlp_input_dim += 7

        # 2. Two-Layer Action Head (MLP)
        # 1024 (or 1031) -> 256 -> 8 (7 joints + 1 gripper)
        self.action_head = nn.Sequential(
            nn.Linear(mlp_input_dim, 256), nn.ReLU(),nn.Dropout(p=0.3), nn.Linear(256, 8)
        )

    def forward(self, obs_t, obs_tk, q_t=None):
        """
        obs_t, obs_tk: Tensors of shape (Batch, 3, H, W). Normalized [0, 1] or standard ImageNet.
        q_t: Tensor of shape (Batch, 7) for joint positions. Required if use_proprio=True.
        """
        # Encode both frames through the EXACT SAME ResNet (Shared Weights)
        # Shape goes from (B, 3, H, W) -> (B, 512, 1, 1)
        feat_t = self.encoder(obs_t)
        feat_tk = self.encoder(obs_tk)

        # Flatten the spatial dimensions: (B, 512, 1, 1) -> (B, 512)
        feat_t = torch.flatten(feat_t, 1)
        feat_tk = torch.flatten(feat_tk, 1)

        # Concatenate the visual features
        features = torch.cat([feat_t, feat_tk], dim=1)  # Shape: (B, 1024)

        # Concat proprioception if the ablation is active
        if self.use_proprio:
            if q_t is None:
                raise ValueError(
                    "Model initialized with use_proprio=True, but q_t was not provided in forward pass."
                )
            features = torch.cat([features, q_t], dim=1)  # Shape: (B, 1031)

        # Predict the realized displacement
        out = self.action_head(features)

        # Split into joint-position change (Δq ∈ R⁷) and gripper change (Δg ∈ R¹)
        delta_q = out[:, :7]
        delta_g = out[:, 7:]

        return delta_q, delta_g

    def freeze_entire_encoder(self):
        """Freezes the entire ResNet backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_top_encoder_layers(self):
        """
        Unfreezes only the final residual blocks of the encoder.
        In our nn.Sequential encoder, indices 6 and 7 are ResNet's layer3 and layer4.
        """
        for i, child in enumerate(self.encoder.children()):
            if i >= 6:
                for param in child.parameters():
                    param.requires_grad = True
