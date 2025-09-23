"""
Utility functions for federated deepfake detection
"""

import torch
import yaml
import logging
import os
import random
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from PIL import Image


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    logging_config = {
        'level': getattr(logging, log_level.upper()),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['handlers'] = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    
    logging.basicConfig(**logging_config)


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_plots(history: Dict[str, List], save_dir: str):
    """
    Save training history plots
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics from history
    rounds = history.get('rounds', [])
    global_metrics = history.get('global_metrics', [])
    
    if not rounds or not global_metrics:
        logging.warning("No training history to plot")
        return
    
    # Extract accuracy and loss values
    accuracies = []
    losses = []
    
    for metrics in global_metrics:
        if isinstance(metrics, dict):
            accuracies.append(metrics.get('global_accuracy', 0))
            losses.append(metrics.get('global_loss', 0))
    
    if not accuracies:
        logging.warning("No metrics found in training history")
        return
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(rounds[:len(accuracies)], accuracies, 'b-', linewidth=2, label='Global Accuracy')
    ax1.set_xlabel('Federated Learning Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Global Model Accuracy Over Rounds')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Loss plot
    ax2.plot(rounds[:len(losses)], losses, 'r-', linewidth=2, label='Global Loss')
    ax2.set_xlabel('Federated Learning Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Global Model Loss Over Rounds')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Training plots saved to {save_dir}")


def evaluate_model_performance(model: torch.nn.Module, 
                             data_loader: torch.utils.data.DataLoader,
                             device: torch.device,
                             save_dir: str = None) -> Dict[str, Any]:
    """
    Evaluate model performance and generate metrics
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        save_dir: Directory to save evaluation plots
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    
    # Classification report
    class_names = ['Real', 'Fake']
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    metrics = {
        'accuracy': accuracy * 100,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }
    
    # Save plots if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curve (for binary classification)
        if len(class_names) == 2:
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(all_targets, np.array(all_probabilities)[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'roc_curve.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            metrics['roc_auc'] = roc_auc
    
    return metrics


def create_sample_data(data_dir: str, num_samples_per_class: int = 50):
    """
    Create sample data for testing the federated learning system
    
    Args:
        data_dir: Directory to create sample data in
        num_samples_per_class: Number of samples per class (real/fake)
    """
    logging.info(f"Creating sample data in {data_dir}")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['real', 'fake']:
            class_dir = os.path.join(data_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Generate synthetic images for demonstration
    image_size = 224
    
    for split in ['train', 'val', 'test']:
        samples_per_class = num_samples_per_class if split == 'train' else num_samples_per_class // 5
        
        for class_idx, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(data_dir, split, class_name)
            
            for i in range(samples_per_class):
                # Generate random image
                if class_name == 'real':
                    # More natural patterns for "real" images
                    image = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
                    # Add some structure
                    image = cv2.GaussianBlur(image, (5, 5), 1.0)
                else:
                    # More artificial patterns for "fake" images
                    image = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
                    # Add noise
                    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
                    image = np.clip(image + noise, 0, 255)
                
                # Save image
                filename = f"{class_name}_{split}_{i:04d}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, image)
    
    logging.info(f"Sample data created successfully in {data_dir}")


def prepare_federated_data(base_data_dir: str, 
                          num_clients: int,
                          distribution: str = 'iid') -> List[str]:
    """
    Prepare data distribution for federated learning clients
    
    Args:
        base_data_dir: Base directory containing the dataset
        num_clients: Number of federated learning clients
        distribution: Data distribution strategy ('iid' or 'non_iid')
    
    Returns:
        List of client data directories
    """
    logging.info(f"Preparing federated data for {num_clients} clients with {distribution} distribution")
    
    client_dirs = []
    
    for client_id in range(num_clients):
        client_data_dir = os.path.join(base_data_dir, f"client_{client_id}")
        
        # Create client directory structure
        for split in ['train', 'val']:
            for class_name in ['real', 'fake']:
                class_dir = os.path.join(client_data_dir, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
        
        client_dirs.append(client_data_dir)
    
    # For demonstration, create sample data for each client
    for client_dir in client_dirs:
        create_sample_data(client_dir, num_samples_per_class=20)
    
    return client_dirs


def visualize_federated_results(training_history: Dict[str, Any], 
                               save_dir: str):
    """
    Create comprehensive visualizations of federated learning results
    
    Args:
        training_history: Complete training history
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot global metrics over rounds
    save_training_plots(training_history, save_dir)
    
    # Plot client participation and performance
    client_metrics = training_history.get('client_metrics', [])
    if client_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Client accuracy over rounds
        client_accuracies = {}
        rounds = range(len(client_metrics))
        
        for round_idx, round_metrics in enumerate(client_metrics):
            for client_id, metrics in round_metrics.items():
                if client_id not in client_accuracies:
                    client_accuracies[client_id] = []
                client_accuracies[client_id].append(metrics.get('accuracy', 0))
        
        for client_id, accuracies in client_accuracies.items():
            ax1.plot(list(rounds)[:len(accuracies)], accuracies, 
                    label=f'Client {client_id}', alpha=0.7)
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Local Accuracy (%)')
        ax1.set_title('Client Local Accuracy Over Rounds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Client training times
        client_times = {}
        for round_idx, round_metrics in enumerate(client_metrics):
            for client_id, metrics in round_metrics.items():
                if client_id not in client_times:
                    client_times[client_id] = []
                client_times[client_id].append(metrics.get('training_time', 0))
        
        for client_id, times in client_times.items():
            ax2.plot(list(rounds)[:len(times)], times, 
                    label=f'Client {client_id}', alpha=0.7)
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Client Training Time Over Rounds')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'client_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Federated learning visualizations saved to {save_dir}")


def model_summary(model: torch.nn.Module, input_size: tuple = (3, 224, 224)):
    """
    Print a summary of the model architecture
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    print(f"\nEstimated model size: {model_size_mb:.2f} MB")
    print("=" * 80)