"""
Federated Learning Server Implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
import os
from collections import defaultdict
import json

try:
    from ..models.deepfake_cnn import create_model
    from .client import ClientManager
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.deepfake_cnn import create_model
    from federated.client import ClientManager


class FederatedServer:
    """
    Federated Learning Server for coordinating deepfake detection training
    """
    
    def __init__(self, config: Dict[str, Any], save_dir: str = "./models"):
        """
        Initialize federated learning server
        
        Args:
            config: Configuration dictionary
            save_dir: Directory to save models and logs
        """
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize global model
        self.global_model = create_model(config['model']).to(self.device)
        
        # Training history
        self.training_history = {
            'rounds': [],
            'global_metrics': [],
            'client_metrics': [],
            'aggregation_weights': []
        }
        
        # Current round
        self.current_round = 0
        
        logging.info("Federated Learning Server initialized")
        logging.info(f"Global model: {config['model']['name']}")
        logging.info(f"Device: {self.device}")
    
    def get_global_parameters(self) -> List[np.ndarray]:
        """Get global model parameters"""
        return [param.detach().cpu().numpy() for param in self.global_model.parameters()]
    
    def set_global_parameters(self, parameters: List[np.ndarray]):
        """Set global model parameters"""
        params_dict = zip(self.global_model.parameters(), parameters)
        for param, new_value in params_dict:
            param.data = torch.tensor(new_value, dtype=param.dtype, device=param.device)
    
    def select_clients(self, client_manager: ClientManager) -> List[int]:
        """
        Select clients for the current round
        
        Args:
            client_manager: Client manager instance
        
        Returns:
            List of selected client IDs
        """
        available_clients = client_manager.get_available_clients()
        min_clients = self.config['federated_learning']['min_fit_clients']
        
        # Simple selection: use all available clients (minimum required)
        if len(available_clients) < min_clients:
            logging.warning(f"Only {len(available_clients)} clients available, "
                          f"but {min_clients} required")
            return [client.client_id for client in available_clients]
        
        # For now, select all available clients
        # In practice, you might implement more sophisticated selection strategies
        selected_clients = [client.client_id for client in available_clients[:min_clients]]
        
        logging.info(f"Selected {len(selected_clients)} clients for round {self.current_round}")
        return selected_clients
    
    def train_round(self, client_manager: ClientManager) -> Dict[str, Any]:
        """
        Execute one round of federated training
        
        Args:
            client_manager: Client manager instance
        
        Returns:
            Dictionary containing round metrics
        """
        round_start_time = time.time()
        
        # Select clients for this round
        selected_client_ids = self.select_clients(client_manager)
        
        if not selected_client_ids:
            raise RuntimeError("No clients selected for training round")
        
        # Broadcast global model to selected clients
        global_params = self.get_global_parameters()
        client_manager.broadcast_parameters(global_params)
        
        # Train clients locally
        client_metrics = {}
        for client_id in selected_client_ids:
            client = client_manager.get_client(client_id)
            
            logging.info(f"Training client {client_id} for round {self.current_round}")
            metrics = client.train()
            client_metrics[client_id] = metrics
        
        # Aggregate model updates
        logging.info("Aggregating model updates from clients")
        aggregated_params = client_manager.aggregate_parameters(selected_client_ids)
        
        # Update global model
        self.set_global_parameters(aggregated_params)
        
        # Evaluate global model on clients
        evaluation_metrics = self.evaluate_global_model(client_manager, selected_client_ids)
        
        round_time = time.time() - round_start_time
        
        round_metrics = {
            'round': self.current_round,
            'selected_clients': selected_client_ids,
            'client_metrics': client_metrics,
            'global_metrics': evaluation_metrics,
            'round_time': round_time
        }
        
        # Update training history
        self.training_history['rounds'].append(self.current_round)
        self.training_history['client_metrics'].append(client_metrics)
        self.training_history['global_metrics'].append(evaluation_metrics)
        
        self.current_round += 1
        
        logging.info(f"Round {self.current_round - 1} completed in {round_time:.2f}s")
        logging.info(f"Global metrics: {evaluation_metrics}")
        
        return round_metrics
    
    def evaluate_global_model(self, 
                             client_manager: ClientManager, 
                             client_ids: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate global model on client data
        
        Args:
            client_manager: Client manager instance
            client_ids: Client IDs to evaluate on (None for all)
        
        Returns:
            Dictionary containing global evaluation metrics
        """
        if client_ids is None:
            client_ids = list(client_manager.clients.keys())
        
        # Broadcast global model to clients
        global_params = self.get_global_parameters()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        client_results = {}
        
        for client_id in client_ids:
            client = client_manager.get_client(client_id)
            
            # Set global model parameters on client
            client.set_parameters(global_params)
            
            # Evaluate on client's validation data
            metrics = client.evaluate()
            
            num_samples = metrics['num_samples']
            if num_samples > 0:
                total_loss += metrics['val_loss'] * num_samples
                total_accuracy += metrics['val_accuracy'] * num_samples
                total_samples += num_samples
                
                client_results[client_id] = metrics
        
        if total_samples == 0:
            return {'global_loss': 0.0, 'global_accuracy': 0.0, 'total_samples': 0}
        
        global_metrics = {
            'global_loss': total_loss / total_samples,
            'global_accuracy': total_accuracy / total_samples,
            'total_samples': total_samples,
            'client_results': client_results
        }
        
        return global_metrics
    
    def train(self, client_manager: ClientManager, num_rounds: int = None) -> Dict[str, Any]:
        """
        Execute federated training for multiple rounds
        
        Args:
            client_manager: Client manager instance
            num_rounds: Number of rounds to train (defaults to config value)
        
        Returns:
            Complete training history
        """
        if num_rounds is None:
            num_rounds = self.config['federated_learning']['rounds']
        
        logging.info(f"Starting federated training for {num_rounds} rounds")
        
        training_start_time = time.time()
        
        for round_num in range(num_rounds):
            try:
                round_metrics = self.train_round(client_manager)
                
                # Save model checkpoint periodically
                if (round_num + 1) % 10 == 0:
                    self.save_checkpoint(round_num + 1)
                
                # Log progress
                if (round_num + 1) % 5 == 0:
                    global_acc = round_metrics['global_metrics']['global_accuracy']
                    logging.info(f"Progress: Round {round_num + 1}/{num_rounds}, "
                               f"Global Accuracy: {global_acc:.2f}%")
                
            except Exception as e:
                logging.error(f"Error in round {round_num}: {e}")
                break
        
        total_training_time = time.time() - training_start_time
        
        # Final evaluation
        final_metrics = self.evaluate_global_model(client_manager)
        
        # Save final model
        self.save_model("final_global_model.pth")
        self.save_training_history()
        
        training_summary = {
            'total_rounds': self.current_round,
            'total_training_time': total_training_time,
            'final_metrics': final_metrics,
            'training_history': self.training_history
        }
        
        logging.info(f"Federated training completed after {self.current_round} rounds")
        logging.info(f"Total time: {total_training_time:.2f}s")
        logging.info(f"Final global accuracy: {final_metrics['global_accuracy']:.2f}%")
        
        return training_summary
    
    def save_model(self, filename: str):
        """Save the global model"""
        model_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config,
            'current_round': self.current_round,
            'training_history': self.training_history
        }, model_path)
        
        logging.info(f"Global model saved to {model_path}")
    
    def load_model(self, filename: str):
        """Load the global model"""
        model_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_round = checkpoint.get('current_round', 0)
        self.training_history = checkpoint.get('training_history', {
            'rounds': [], 'global_metrics': [], 'client_metrics': [], 'aggregation_weights': []
        })
        
        logging.info(f"Global model loaded from {model_path}")
    
    def save_checkpoint(self, round_num: int):
        """Save a checkpoint of the current state"""
        checkpoint_filename = f"checkpoint_round_{round_num}.pth"
        self.save_model(checkpoint_filename)
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history_path = os.path.join(self.save_dir, "training_history.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                serializable_history[key] = value
            else:
                serializable_history[key] = str(value)
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logging.info(f"Training history saved to {history_path}")
    
    def get_model_for_inference(self) -> nn.Module:
        """Get the global model for inference"""
        self.global_model.eval()
        return self.global_model
    
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the global model
        
        Args:
            input_tensor: Input tensor for prediction
        
        Returns:
            Model predictions
        """
        self.global_model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.global_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities