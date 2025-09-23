"""
Deepfake Detection Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DeepfakeDetectionCNN(nn.Module):
    """
    CNN model for deepfake detection with feature extraction capabilities
    suitable for federated learning scenarios.
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(DeepfakeDetectionCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Classification
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head"""
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        
        return x


class ImprovedDeepfakeDetectionCNN(nn.Module):
    """
    Improved CNN model with attention mechanism for better deepfake detection
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(ImprovedDeepfakeDetectionCNN, self).__init__()
        
        # Feature extraction with residual connections
        self.conv_block1 = self._make_conv_block(3, 64)
        self.conv_block2 = self._make_conv_block(64, 128)
        self.conv_block3 = self._make_conv_block(128, 256)
        self.conv_block4 = self._make_conv_block(256, 512)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block with batch norm and max pooling"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism"""
        # Feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_model(config: dict) -> nn.Module:
    """Factory function to create model based on configuration"""
    model_name = config.get('name', 'DeepfakeDetectionCNN')
    num_classes = config.get('num_classes', 2)
    dropout_rate = config.get('dropout_rate', 0.5)
    
    if model_name == 'DeepfakeDetectionCNN':
        return DeepfakeDetectionCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'ImprovedDeepfakeDetectionCNN':
        return ImprovedDeepfakeDetectionCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model name: {model_name}")