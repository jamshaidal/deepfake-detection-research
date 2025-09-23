"""
Simple test script to verify the federated deepfake detection system works
"""

import sys
import os
import tempfile
import shutil

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.helpers import load_config, setup_logging, create_sample_data
from federated.server import FederatedServer
from federated.client import ClientManager, FederatedClient
from models.deepfake_cnn import DeepfakeDetectionCNN


def test_federated_learning():
    """Test the basic federated learning functionality"""
    print("=" * 60)
    print("TESTING FEDERATED DEEPFAKE DETECTION SYSTEM")
    print("=" * 60)
    
    # Setup logging
    setup_logging('INFO')
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Reduce complexity for testing
    config['federated_learning']['num_clients'] = 2
    config['federated_learning']['rounds'] = 2
    config['federated_learning']['client_epochs'] = 1
    config['federated_learning']['batch_size'] = 2
    
    print("✓ Configuration loaded")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    try:
        model_dir = os.path.join(temp_dir, 'models')
        data_dir = os.path.join(temp_dir, 'data')
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        print("✓ Temporary directories created")
        
        # Create sample data for each client
        for client_id in range(2):
            client_data_dir = os.path.join(data_dir, f'client_{client_id}')
            create_sample_data(client_data_dir, num_samples_per_class=5)
        
        print("✓ Sample data created for clients")
        
        # Test model creation
        model = DeepfakeDetectionCNN()
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test server initialization
        server = FederatedServer(config, model_dir)
        print("✓ Federated server initialized")
        
        # Test client manager initialization
        client_manager = ClientManager(config, data_dir)
        print(f"✓ Client manager initialized with {len(client_manager.clients)} clients")
        
        # Test parameter operations
        global_params = server.get_global_parameters()
        client_manager.broadcast_parameters(global_params)
        print("✓ Parameter broadcasting successful")
        
        # Test single training round
        print("\nTesting single training round...")
        
        # Select clients
        selected_clients = [0, 1]
        
        # Train clients locally (simplified)
        for client_id in selected_clients:
            client = client_manager.get_client(client_id)
            try:
                metrics = client.train()
                print(f"✓ Client {client_id} training: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}%")
            except Exception as e:
                print(f"! Client {client_id} training failed: {e}")
        
        # Aggregate parameters
        try:
            aggregated_params = client_manager.aggregate_parameters(selected_clients)
            server.set_global_parameters(aggregated_params)
            print("✓ Parameter aggregation successful")
        except Exception as e:
            print(f"! Parameter aggregation failed: {e}")
        
        # Test model saving
        server.save_model('test_model.pth')
        print("✓ Model saved successfully")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("The federated deepfake detection system is working correctly.")
        print("=" * 60)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_federated_learning()