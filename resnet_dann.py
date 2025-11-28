"""
Domain Adversarial Neural Network (DANN) for Cell Classification

This architecture addresses the donor bias problem:
- D1-D3, D6: Small cells, mostly inactive (70-77% inactive)
- D4-D5: Large cells, mostly active (67-74% active)

The model learns activity features while being invariant to donor-specific
features (cell size, texture) through adversarial training.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    "Domain-Adversarial Training of Neural Networks" (Ganin & Lempitsky, 2015)
    
    Forward pass: Identity (x → x)
    Backward pass: Reverses gradients (∂L/∂x → -λ * ∂L/∂x)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for GradientReversalFunction"""
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ============================================================================
# RESIDUAL BLOCK (Same as base ResNet)
# ============================================================================
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu(fx)
        fx = self.conv2(fx)
        fx = self.bn2(fx)
        out = self.relu(fx + self.shortcut(x))
        return out


# ============================================================================
# DANN RESNET3D
# ============================================================================
class ResNet3D_DANN(nn.Module):
    """
    ResNet3D with Domain Adversarial Training
    
    Architecture:
    ┌─────────────────────────────────────────┐
    │         Feature Extractor               │
    │  (Shared backbone - ResNet3D blocks)    │
    └─────────────┬───────────────────────────┘
                  │ features (64-dim)
                  │
         ┌────────┴──────────┐
         │                   │
    ┌────▼────────┐   ┌──────▼────────────┐
    │  Activity   │   │  Gradient         │
    │  Classifier │   │  Reversal Layer   │
    │             │   │  (λ = 0.1)        │
    │  Active/    │   │                   │
    │  Inactive   │   │  ┌────▼────────┐  │
    └─────────────┘   │  │  Donor      │  │
                      │  │  Classifier │  │
                      │  │  (D1-D6)    │  │
                      │  └─────────────┘  │
                      └───────────────────┘
    
    Training:
    - Activity head: Learns to predict active/inactive (main task)
    - Donor head: Tries to predict which donor (D1-D6)
    - GRL: Reverses gradients from donor head
    - Result: Features good for activity, bad for donor prediction
    """
    
    def __init__(self, dropout_rate=0.3, in_channels=2, num_donors=6, lambda_grl=0.1):
        super(ResNet3D_DANN, self).__init__()
        
        self.num_donors = num_donors
        self.lambda_grl = lambda_grl
        
        # ====================================================================
        # SHARED FEATURE EXTRACTOR (Backbone)
        # ====================================================================
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2))
        )
        
        self.res_block1 = ResBlock3D(in_channels=32, out_channels=32)
        self.res_block2 = ResBlock3D(in_channels=32, out_channels=32)
        self.pool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2))
        self.dropout1 = nn.Dropout3d(p=dropout_rate)
        
        self.res_block3 = ResBlock3D(in_channels=32, out_channels=64)
        self.res_block4 = ResBlock3D(in_channels=64, out_channels=64)
        self.dropout2 = nn.Dropout3d(p=dropout_rate)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # ====================================================================
        # ACTIVITY CLASSIFIER (Main task)
        # ====================================================================
        self.activity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1)  # Binary: active/inactive
        )
        
        # ====================================================================
        # DONOR CLASSIFIER (Adversarial task)
        # ====================================================================
        self.gradient_reversal = GradientReversalLayer(lambda_=lambda_grl)
        self.donor_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, num_donors)  # Multi-class: D1, D2, D3, D4, D5, D6
        )
    
    def extract_features(self, x):
        """Extract shared features (for both activity and donor classification)"""
        # Handle input format
        if x.dim() == 6:
            x = x.squeeze(0)
        if x.shape[-1] in [1, 2]:
            x = x.permute(0, 4, 3, 1, 2)
        
        # Feature extraction
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.dropout2(x)
        
        x = self.global_pool(x)
        features = torch.flatten(x, 1)  # (batch, 64)
        
        return features
    
    def forward(self, x, return_donor_pred=False):
        """
        Args:
            x: Input tensor (batch, 2, 256, 21, 21)
            return_donor_pred: If True, return (activity_pred, donor_pred)
                              If False, return activity_pred only
        
        Returns:
            activity_pred: Active/Inactive prediction (batch, 1)
            donor_pred: Donor prediction (batch, 6) - only if return_donor_pred=True
        """
        # Extract features
        features = self.extract_features(x)
        
        # Activity classification (main task)
        activity_pred = self.activity_head(features)
        
        if return_donor_pred:
            # Donor classification with gradient reversal
            reversed_features = self.gradient_reversal(features)
            donor_pred = self.donor_head(reversed_features)
            return activity_pred, donor_pred
        else:
            return activity_pred


# ============================================================================
# MODEL CREATION FUNCTION
# ============================================================================
def create_dann_model(in_channels=2, dropout_rate=0.25, num_donors=6, lambda_grl=0.1):
    """
    Create a DANN model for domain adversarial training
    
    Args:
        in_channels: Number of input channels (2 for decay + intensity)
        dropout_rate: Dropout probability
        num_donors: Number of donor classes (6 for D1-D6)
        lambda_grl: Gradient reversal strength (0.1 is typical)
    
    Returns:
        ResNet3D_DANN model
    """
    return ResNet3D_DANN(
        dropout_rate=dropout_rate,
        in_channels=in_channels,
        num_donors=num_donors,
        lambda_grl=lambda_grl
    )


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_dann_model(in_channels=2).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 2, 256, 21, 21).to(device)
    
    # Activity prediction only
    activity_pred = model(x, return_donor_pred=False)
    print(f"\nActivity prediction shape: {activity_pred.shape}")  # (2, 1)
    
    # Both predictions
    activity_pred, donor_pred = model(x, return_donor_pred=True)
    print(f"Activity prediction shape: {activity_pred.shape}")  # (2, 1)
    print(f"Donor prediction shape: {donor_pred.shape}")  # (2, 6)
    
    print("\n✓ DANN model created successfully!")
