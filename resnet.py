import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary

# Only import device if prepare_data is available (for backward compatibility)
try:
    from train import device
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define the Residual Block --- #
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock3D, self).__init__()

        # Path A: Convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Path B: Skip Connection
        self.shortcut = nn.Sequential()

        # if in != out, we need 1x1 convolution to match them
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
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
    def __init__(self):
        super(ResNet3D, self).__init__()

        # Initial Convolution Layer
        self.initial_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Stack of Residual Blocks
        self.res_block1 = ResBlock3D(in_channels=32, out_channels=32)
        self.res_block2 = ResBlock3D(in_channels=32, out_channels=32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
    
        self.res_block3 = ResBlock3D(in_channels=32, out_channels=64)
        self.res_block4 = ResBlock3D(in_channels=64, out_channels=64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.res_block5 = ResBlock3D(in_channels=64, out_channels=128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # PyTorch expects (N, C, D, H, W)
        x = x.permute(0, 4, 3, 1, 2)

        # Initial Convolution Layer
        x = self.initial_conv(x)
        
        # Stack of Residual Blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool1(x)
        
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.pool2(x)

        x = self.res_block5(x)
        x = self.pool3(x)

        # Global Average Pooling
        x = self.global_pool(x)

        # Flatten the output for the classifier
        x = torch.flatten(x, 1)

        # Classify
        x = self.classifier(x)

        return x


# --- Function to create a fresh model instance --- #
def create_model():
    """
    Factory function to create a new ResNet3D model instance.
    Useful for cross-validation where each fold needs a fresh model.
    """
    return ResNet3D()

    
# --- Instantiate the model, loss, and optimizer (for backward compatibility) ---
model = ResNet3D().to(device)
loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss 
optimizer = optim.Adam(model.parameters(), lr=0.001)

summary(model, input_size=(1, 32, 32, 32))



    
    

        
        

