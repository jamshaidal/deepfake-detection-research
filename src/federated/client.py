"""
Federated Learning Client Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any
import copy
import logging
from collections import OrderedDict
import time

try:
    from ..models.deepfake_cnn import create_model
    from ..data.preprocessing import create_data_loaders
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.deepfake_cnn import create_model
    from data.preprocessing import create_data_loaders


class FederatedClient:
    """
    Federated Learning Client for deepfake detection
    """
    
    def __init__(self, 
                 client_id: int,
                 config: Dict[str, Any],
                 data_dir: str,
                 device: torch.device = None):
        """
        Initialize federated learning client
        
        Args:
            client_id: Unique identifier for this client
            config: Configuration dictionary
            data_dir: Directory containing client's local data
            device: Device to run computations on
        """
        self.client_id = client_id
        self.config = config
        self.data_dir = data_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = create_model(config['model']).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['federated_learning']['learning_rate']
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Load local data
        self._load_local_data()
        
        # Training metrics
        self.training_history = []
        
        logging.info(f"Client {client_id} initialized with {len(self.train_loader.dataset)} training samples")
    
    def _load_local_data(self):
        """Load local training and validation data"""
        try:
            self.train_loader, self.val_loader = create_data_loaders(
                data_dir=self.data_dir,
                batch_size=self.config['federated_learning']['batch_size'],
                image_size=self.config['data']['image_size'],
                augmentation=self.config['data']['augmentation'],
                num_workers=self.config['training']['num_workers'],
                pin_memory=self.config['training']['pin_memory']
            )
        except Exception as e:
            logging.warning(f"Client {self.client_id}: Failed to load data, creating dummy loaders: {e}")
            # Create dummy data loaders for demonstration
            from torch.utils.data import TensorDataset
            dummy_data = torch.randn(10, 3, 224, 224)
            dummy_labels = torch.randint(0, 2, (10,))
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            self.train_loader = DataLoader(dummy_dataset, batch_size=4)
            self.val_loader = DataLoader(dummy_dataset, batch_size=4)
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays"""
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_value in params_dict:
            param.data = torch.tensor(new_value, dtype=param.dtype, device=param.device)
    
    def train(self, epochs: int = None) -> Dict[str, float]:
        """
        Train the model locally
        
        Args:
            epochs: Number of epochs to train (defaults to config value)
        
        Returns:
            Dictionary containing training metrics
        """
        if epochs is None:
            epochs = self.config['federated_learning']['client_epochs']
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += targets.size(0)
                epoch_correct += (predicted == targets).sum().item()
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
            
            logging.debug(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}: "
                         f"Loss: {epoch_loss/len(self.train_loader):.4f}, "
                         f"Accuracy: {100*epoch_correct/epoch_total:.2f}%")
        
        training_time = time.time() - start_time
        avg_loss = total_loss / (epochs * len(self.train_loader))
        accuracy = 100 * correct / total
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'training_time': training_time,
            'num_samples': total
        }
        
        self.training_history.append(metrics)
        
        logging.info(f"Client {self.client_id} training completed: "
                    f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
                    f"Time: {training_time:.2f}s")
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'num_samples': total
        }
        
        logging.info(f"Client {self.client_id} evaluation: "
                    f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return metrics
    
    def get_num_samples(self) -> int:
        """Get number of training samples"""
        return len(self.train_loader.dataset)
    
    def save_model(self, path: str):
        """Save the model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'client_id': self.client_id
        }, path)
        
        logging.info(f"Client {self.client_id} model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        
        logging.info(f"Client {self.client_id} model loaded from {path}")


class ClientManager:
    """
    Manager class for handling multiple federated learning clients
    """
    
    def __init__(self, config: Dict[str, Any], data_base_dir: str):
        """
        Initialize client manager
        
        Args:
            config: Configuration dictionary
            data_base_dir: Base directory containing client data directories
        """
        self.config = config
        self.data_base_dir = data_base_dir
        self.clients = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all federated learning clients"""
        num_clients = self.config['federated_learning']['num_clients']
        
        for client_id in range(num_clients):
            client_data_dir = f"{self.data_base_dir}/client_{client_id}"
            
            client = FederatedClient(
                client_id=client_id,
                config=self.config,
                data_dir=client_data_dir,
                device=self.device
            )
            
            self.clients[client_id] = client
        
        logging.info(f"Initialized {len(self.clients)} federated learning clients")
    
    def get_client(self, client_id: int) -> FederatedClient:
        """Get a specific client"""
        return self.clients.get(client_id)
    
    def get_all_clients(self) -> List[FederatedClient]:
        """Get all clients"""
        return list(self.clients.values())
    
    def get_available_clients(self) -> List[FederatedClient]:
        """Get all available clients (simplified - all clients are always available)"""
        return self.get_all_clients()
    
    def broadcast_parameters(self, parameters: List[np.ndarray]):
        """Broadcast global model parameters to all clients"""
        for client in self.clients.values():
            client.set_parameters(parameters)
        
        logging.info(f"Broadcasted parameters to {len(self.clients)} clients")
    
    def aggregate_parameters(self, client_ids: List[int]) -> List[np.ndarray]:
        """
        Aggregate parameters from specified clients using FedAvg
        
        Args:
            client_ids: List of client IDs to aggregate from
        
        Returns:
            Aggregated parameters
        """
        if not client_ids:
            raise ValueError("No clients specified for aggregation")
        
        # Get parameters and sample counts from each client
        client_parameters = []
        sample_counts = []
        
        for client_id in client_ids:
            client = self.clients[client_id]
            client_parameters.append(client.get_parameters())
            sample_counts.append(client.get_num_samples())
        
        # Weighted averaging based on number of samples
        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts]
        
        # Aggregate parameters
        aggregated_params = []
        for param_idx in range(len(client_parameters[0])):
            weighted_param = np.zeros_like(client_parameters[0][param_idx])
            
            for client_idx, weight in enumerate(weights):
                weighted_param += weight * client_parameters[client_idx][param_idx]
            
            aggregated_params.append(weighted_param)
        
        logging.info(f"Aggregated parameters from {len(client_ids)} clients "
                    f"with total {total_samples} samples")
        
        return aggregated_params