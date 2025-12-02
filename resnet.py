import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
try:
    from torchsummary import summary
except ImportError:
    summary = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define the Residual Block --- #
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, temporal_stride=1):
        super(ResBlock3D, self).__init__()

        # FIXED: Separate temporal and spatial strides to prevent double downsampling
        # stride applies to (H, W), temporal_stride applies to T dimension
        if isinstance(stride, int):
            stride_3d = (temporal_stride, stride, stride)  # (T, H, W)
        else:
            stride_3d = stride

        # Path A: Convolution (stride ONLY in first conv)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride_3d, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)  # stride=1!
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Path B: Skip Connection
        self.shortcut = nn.Sequential()

        # Adjust shortcut if channels or spatial dims change
        if in_channels != out_channels or stride_3d != (1, 1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride_3d),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        # Path A
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu(fx)
        fx = self.conv2(fx)
        fx = self.bn2(fx)

        # Add Paths A and B
        out = self.relu(fx + self.shortcut(x))
        return out
    
    
# --- Define the Full 3D ResNet --- #
class ResNet3D(nn.Module):
    def __init__(self, dropout_rate=0.3, in_channels=2):  # 2 channels: FLIM + Mask
        super(ResNet3D, self).__init__()

        # STEM: Temporal feature extraction (Matched to baseline F1=0.91 model)
        # Use (7,1,1) kernel to prioritize temporal decay curve learning
        # Input: (2, 256, 21, 21) -> Output: (32, 256, 21, 21)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # Gentle Temporal Downsampling: 256 -> 128 temporal bins
        # Preserve spatial resolution (21x21)
        self.downsample1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # STAGE 1: High resolution processing (128 temporal, 21×21 spatial)
        # NO DROPOUT - BatchNorm provides sufficient regularization (standard ResNet practice)
        self.res_block1 = ResBlock3D(32, 32, stride=1, temporal_stride=1)

        # STAGE 2: Spatial downsampling (21 → 11), Temporal: 128 → 64
        self.res_block2 = ResBlock3D(32, 64, stride=2, temporal_stride=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        # STAGE 3: Final downsampling (11 → 6), Temporal: 64 → 32
        self.res_block3 = ResBlock3D(64, 128, stride=2, temporal_stride=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Classifier (dropout ONLY here, not between ResBlocks)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=dropout_rate),  # Only dropout before final layer
            nn.Linear(64, 1)
        )
        
        # Initialize weights with Kaiming Normal (He Init) for deep 3D networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Kaiming Normal for stable deep network training."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle torchsummary adding extra dimension
        if x.dim() == 6:
            x = x.squeeze(0)
        
        # PyTorch Conv3d expects (N, C, D, H, W)
        # Our dataset returns (N, T, H, W) for single channel
        # Add channel dimension if missing
        if x.dim() == 4:  # (N, T, H, W) -> (N, 1, T, H, W)
            x = x.unsqueeze(1)
        
        # If input has wrong channel position, fix it
        # Legacy check for old 2-channel format: (N, H, W, T, C)
        if x.dim() == 5 and x.shape[-1] in [1, 2]:
            x = x.permute(0, 4, 3, 1, 2)  # (N, H, W, T, C) -> (N, C, T, H, W)

        # Spatiotemporal Stem + Downsampling
        x = self.stem(x)          # (2, 256, 21, 21) -> (32, 256, 21, 21)
        x = self.downsample1(x)   # (32, 256, 21, 21) -> (32, 128, 21, 21)
        
        # Stage 1: High resolution processing
        x = self.res_block1(x)    # (32, 128, 21, 21)

        # Stage 2: Spatial + temporal downsampling
        x = self.res_block2(x)    # Spatial: 21 → 11
        x = self.pool2(x)         # Temporal: 128 → 64

        # Stage 3: Final downsampling
        x = self.res_block3(x)    # Spatial: 11 → 6
        x = self.pool3(x)         # Temporal: 64 → 32

        # Global Average Pooling
        x = self.global_pool(x)

        # Flatten the output for the classifier
        x = torch.flatten(x, 1)

        # Classify
        x = self.classifier(x)

        return x


# --- Function to create a fresh model instance --- #
def create_model(in_channels=2, dropout_rate=0.25):  # 2 channels: FLIM + mask
    return ResNet3D(dropout_rate=dropout_rate, in_channels=in_channels)


# --- Main block to avoid import errors --- #
if __name__ == "__main__":
    # Only instantiate model when running this file directly (for testing)
    model = ResNet3D(in_channels=2, dropout_rate=0.3).to(device)  # 2 channels: FLIM + Mask
    loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits (correct for raw logits)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if summary is not None:
        summary(model, input_size=(2, 256, 21, 21))  # (C, T, H, W) - 2 channels
    else:
        print("torchsummary not installed, skipping model summary")



    
    

        
        

