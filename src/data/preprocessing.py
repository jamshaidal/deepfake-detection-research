"""
Data preprocessing utilities for deepfake detection
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple, Optional, Union
import logging

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition not available. Face extraction will be disabled.")


class DeepfakeDataset(Dataset):
    """
    Dataset class for deepfake detection with face extraction and preprocessing
    """
    
    def __init__(self, 
                 data_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 extract_faces: bool = True,
                 image_size: int = 224):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing real and fake subdirectories
            transform: Image transformations to apply
            extract_faces: Whether to extract faces from images
            image_size: Target image size for resizing
        """
        self.data_dir = data_dir
        self.transform = transform
        self.extract_faces = extract_faces
        self.image_size = image_size
        self.samples = []
        self.labels = []
        
        self._load_data()
        
    def _load_data(self):
        """Load data samples and labels from directory structure"""
        # Expected structure: data_dir/real/, data_dir/fake/
        real_dir = os.path.join(self.data_dir, 'real')
        fake_dir = os.path.join(self.data_dir, 'fake')
        
        # Load real images (label 0)
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(real_dir, filename))
                    self.labels.append(0)
        
        # Load fake images (label 1)
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(fake_dir, filename))
                    self.labels.append(1)
        
        logging.info(f"Loaded {len(self.samples)} samples ({sum(1 for l in self.labels if l == 0)} real, "
                    f"{sum(1 for l in self.labels if l == 1)} fake)")
    
    def _extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract the largest face from the image"""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
            
        try:
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) > 0:
                # Get the largest face
                largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
                top, right, bottom, left = largest_face
                
                # Extract face with some padding
                padding = 20
                top = max(0, top - padding)
                left = max(0, left - padding)
                bottom = min(image.shape[0], bottom + padding)
                right = min(image.shape[1], right + padding)
                
                face = image[top:bottom, left:right]
                return face
            else:
                return None
        except Exception as e:
            logging.warning(f"Face extraction failed: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample and its label"""
        image_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Create a dummy image if loading fails
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract face if requested
            if self.extract_faces:
                face = self._extract_face(image)
                if face is not None:
                    image = face
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = transforms.ToTensor()(image)
        
        return image, label


def get_transforms(image_size: int = 224, augmentation: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms
    
    Args:
        image_size: Target image size
        augmentation: Whether to apply data augmentation for training
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Base transforms
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    # Training transforms with augmentation
    if augmentation:
        train_transforms = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
    else:
        train_transforms = base_transforms
    
    # Validation transforms (no augmentation)
    val_transforms = base_transforms
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def create_data_loaders(data_dir: str, 
                       batch_size: int = 32,
                       image_size: int = 224,
                       augmentation: bool = True,
                       extract_faces: bool = True,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_dir: Directory containing train/val subdirectories
        batch_size: Batch size for data loaders
        image_size: Target image size
        augmentation: Whether to apply data augmentation
        extract_faces: Whether to extract faces
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform, val_transform = get_transforms(image_size, augmentation)
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=os.path.join(data_dir, 'train'),
        transform=train_transform,
        extract_faces=extract_faces,
        image_size=image_size
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=os.path.join(data_dir, 'val'),
        transform=val_transform,
        extract_faces=extract_faces,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def prepare_client_data(data_dir: str, 
                       num_clients: int,
                       train_split: float = 0.8,
                       val_split: float = 0.1) -> List[Tuple[str, str]]:
    """
    Prepare data distribution for federated learning clients
    
    Args:
        data_dir: Directory containing real and fake subdirectories
        num_clients: Number of federated learning clients
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
    
    Returns:
        List of (train_dir, val_dir) tuples for each client
    """
    # This is a simplified implementation
    # In practice, you would implement more sophisticated data distribution strategies
    client_dirs = []
    
    for client_id in range(num_clients):
        client_train_dir = os.path.join(data_dir, f'client_{client_id}', 'train')
        client_val_dir = os.path.join(data_dir, f'client_{client_id}', 'val')
        
        os.makedirs(client_train_dir, exist_ok=True)
        os.makedirs(client_val_dir, exist_ok=True)
        
        client_dirs.append((client_train_dir, client_val_dir))
    
    return client_dirs


def video_to_frames(video_path: str, 
                   output_dir: str, 
                   max_frames: int = 100,
                   skip_frames: int = 5) -> List[str]:
    """
    Extract frames from video for deepfake detection
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract
        skip_frames: Number of frames to skip between extractions
    
    Returns:
        List of extracted frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % (skip_frames + 1) == 0:
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    return frame_paths