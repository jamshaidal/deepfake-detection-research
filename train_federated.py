"""
Main training script for federated deepfake detection
"""

import argparse
import os
import sys
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.helpers import load_config, setup_logging, set_random_seeds, create_sample_data, prepare_federated_data
from federated.server import FederatedServer
from federated.client import ClientManager


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Federated Learning for Deepfake Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Path to save models')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Number of federated learning rounds (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    setup_logging(args.log_level, 'logs/federated_training.log')
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return
    
    # Override config with command line arguments
    if args.rounds is not None:
        config['federated_learning']['rounds'] = args.rounds
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create sample data if requested
    if args.create_sample_data:
        logging.info("Creating sample data for demonstration...")
        create_sample_data(os.path.join(args.data_dir, 'base'), num_samples_per_class=100)
        
        # Prepare federated data distribution
        prepare_federated_data(
            args.data_dir, 
            config['federated_learning']['num_clients']
        )
        logging.info("Sample data created successfully")
    
    # Initialize federated learning server
    logging.info("Initializing Federated Learning Server...")
    server = FederatedServer(config, args.model_dir)
    
    # Initialize client manager
    logging.info("Initializing Client Manager...")
    client_manager = ClientManager(config, args.data_dir)
    
    # Start federated training
    logging.info("Starting federated training...")
    try:
        training_summary = server.train(client_manager)
        
        # Log final results
        final_accuracy = training_summary['final_metrics']['global_accuracy']
        total_time = training_summary['total_training_time']
        total_rounds = training_summary['total_rounds']
        
        logging.info("=" * 50)
        logging.info("FEDERATED TRAINING COMPLETED")
        logging.info("=" * 50)
        logging.info(f"Total Rounds: {total_rounds}")
        logging.info(f"Final Global Accuracy: {final_accuracy:.2f}%")
        logging.info(f"Total Training Time: {total_time:.2f} seconds")
        logging.info(f"Average Time per Round: {total_time/total_rounds:.2f} seconds")
        logging.info("=" * 50)
        
        # Save visualizations
        from utils.helpers import visualize_federated_results
        visualize_federated_results(training_summary['training_history'], 
                                   os.path.join(args.model_dir, 'visualizations'))
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()