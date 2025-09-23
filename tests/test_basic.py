"""
Basic tests for the federated deepfake detection system
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.deepfake_cnn import DeepfakeDetectionCNN, create_model
from federated.client import FederatedClient
from federated.server import FederatedServer
from utils.helpers import load_config, create_sample_data


class TestModels(unittest.TestCase):
    """Test model creation and basic functionality"""
    
    def test_model_creation(self):
        """Test that models can be created and forward pass works"""
        model = DeepfakeDetectionCNN(num_classes=2, dropout_rate=0.5)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 2))
        self.assertFalse(torch.isnan(output).any())
    
    def test_model_factory(self):
        """Test model factory function"""
        config = {
            'name': 'DeepfakeDetectionCNN',
            'num_classes': 2,
            'dropout_rate': 0.3
        }
        
        model = create_model(config)
        self.assertIsInstance(model, DeepfakeDetectionCNN)
    
    def test_feature_extraction(self):
        """Test feature extraction functionality"""
        model = DeepfakeDetectionCNN()
        dummy_input = torch.randn(2, 3, 224, 224)
        
        features = model.extract_features(dummy_input)
        self.assertEqual(features.shape[0], 2)  # Batch size
        self.assertTrue(features.shape[1] > 0)  # Feature dimension


class TestFederatedLearning(unittest.TestCase):
    """Test federated learning components"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'federated_learning': {
                'num_clients': 2,
                'rounds': 2,
                'client_epochs': 1,
                'batch_size': 4,
                'learning_rate': 0.001,
                'min_fit_clients': 1,
                'min_eval_clients': 1,
                'min_available_clients': 1
            },
            'model': {
                'name': 'DeepfakeDetectionCNN',
                'num_classes': 2,
                'dropout_rate': 0.5
            },
            'data': {
                'image_size': 224,
                'augmentation': False
            },
            'training': {
                'num_workers': 0,
                'pin_memory': False
            }
        }
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_federated_server_initialization(self):
        """Test that federated server can be initialized"""
        server = FederatedServer(self.config, self.temp_dir)
        
        self.assertIsNotNone(server.global_model)
        self.assertEqual(server.current_round, 0)
    
    def test_parameter_operations(self):
        """Test parameter get/set operations"""
        server = FederatedServer(self.config, self.temp_dir)
        
        # Get parameters
        params = server.get_global_parameters()
        self.assertIsInstance(params, list)
        self.assertTrue(len(params) > 0)
        
        # Set parameters
        server.set_global_parameters(params)
        
        # Verify parameters are same
        new_params = server.get_global_parameters()
        for p1, p2 in zip(params, new_params):
            np.testing.assert_array_equal(p1, p2)
    
    def test_client_initialization(self):
        """Test that federated client can be initialized"""
        client = FederatedClient(
            client_id=0,
            config=self.config,
            data_dir=self.temp_dir
        )
        
        self.assertEqual(client.client_id, 0)
        self.assertIsNotNone(client.model)
        self.assertIsNotNone(client.optimizer)


class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_sample_data_creation(self):
        """Test sample data creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            create_sample_data(temp_dir, num_samples_per_class=5)
            
            # Check if directories were created
            train_real = os.path.join(temp_dir, 'train', 'real')
            train_fake = os.path.join(temp_dir, 'train', 'fake')
            
            self.assertTrue(os.path.exists(train_real))
            self.assertTrue(os.path.exists(train_fake))
            
            # Check if images were created
            real_images = os.listdir(train_real)
            fake_images = os.listdir(train_fake)
            
            self.assertEqual(len(real_images), 5)
            self.assertEqual(len(fake_images), 5)


if __name__ == '__main__':
    unittest.main()