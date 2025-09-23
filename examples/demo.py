"""
Example script demonstrating the complete federated learning workflow
"""

import os
import sys
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.helpers import (
    load_config, setup_logging, set_random_seeds, create_sample_data, 
    prepare_federated_data, visualize_federated_results, model_summary
)
from federated.server import FederatedServer
from federated.client import ClientManager
from models.deepfake_cnn import create_model


def demo_federated_learning():
    """
    Demonstrate the complete federated learning workflow for deepfake detection
    """
    print("=" * 80)
    print("FEDERATED DEEPFAKE DETECTION DEMO")
    print("=" * 80)
    
    # Setup
    setup_logging('INFO')
    set_random_seeds(42)
    
    # Load configuration
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found!")
        return
    
    config = load_config(config_path)
    
    # Create directories
    demo_dir = './demo_results'
    data_dir = os.path.join(demo_dir, 'data')
    model_dir = os.path.join(demo_dir, 'models')
    
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Demo directory: {demo_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    
    # Step 1: Create sample data
    print("\n" + "="*50)
    print("STEP 1: CREATING SAMPLE DATA")
    print("="*50)
    
    print("Creating sample deepfake dataset...")
    create_sample_data(os.path.join(data_dir, 'base'), num_samples_per_class=50)
    
    print("Preparing federated data distribution...")
    client_dirs = prepare_federated_data(
        data_dir, 
        config['federated_learning']['num_clients']
    )
    
    print(f"Created data for {len(client_dirs)} clients")
    
    # Step 2: Initialize federated learning components
    print("\n" + "="*50)
    print("STEP 2: INITIALIZING FEDERATED LEARNING")
    print("="*50)
    
    print("Creating global model...")
    global_model = create_model(config['model'])
    model_summary(global_model)
    
    print("\nInitializing Federated Learning Server...")
    server = FederatedServer(config, model_dir)
    
    print("Initializing Client Manager...")
    client_manager = ClientManager(config, data_dir)
    
    print(f"Federated setup complete with {len(client_manager.clients)} clients")
    
    # Step 3: Run federated training (shortened for demo)
    print("\n" + "="*50)
    print("STEP 3: FEDERATED TRAINING")
    print("="*50)
    
    # Reduce rounds for demo
    demo_rounds = min(5, config['federated_learning']['rounds'])
    config['federated_learning']['rounds'] = demo_rounds
    
    print(f"Starting federated training for {demo_rounds} rounds...")
    
    try:
        training_summary = server.train(client_manager, num_rounds=demo_rounds)
        
        print("\nTraining completed successfully!")
        print(f"Final global accuracy: {training_summary['final_metrics']['global_accuracy']:.2f}%")
        print(f"Total training time: {training_summary['total_training_time']:.2f} seconds")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Generate visualizations
    print("\n" + "="*50)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*50)
    
    viz_dir = os.path.join(demo_dir, 'visualizations')
    visualize_federated_results(training_summary['training_history'], viz_dir)
    print(f"Visualizations saved to {viz_dir}")
    
    # Step 5: Model evaluation
    print("\n" + "="*50)
    print("STEP 5: MODEL EVALUATION")
    print("="*50)
    
    final_model = server.get_model_for_inference()
    
    # Evaluate on a client's data
    client = client_manager.get_client(0)
    client.set_parameters(server.get_global_parameters())
    
    eval_metrics = client.evaluate()
    print(f"Evaluation on Client 0:")
    print(f"  Validation Accuracy: {eval_metrics['val_accuracy']:.2f}%")
    print(f"  Validation Loss: {eval_metrics['val_loss']:.4f}")
    
    # Step 6: Save final model
    print("\n" + "="*50)
    print("STEP 6: SAVING FINAL MODEL")
    print("="*50)
    
    final_model_path = os.path.join(model_dir, 'final_federated_model.pth')
    server.save_model('final_federated_model.pth')
    print(f"Final model saved to: {final_model_path}")
    
    # Demo summary
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved in: {demo_dir}")
    print("\nWhat was demonstrated:")
    print("✓ Federated learning setup with multiple clients")
    print("✓ Distributed training of deepfake detection model")
    print("✓ Model aggregation using FedAvg algorithm")
    print("✓ Performance evaluation and visualization")
    print("✓ Model saving for future inference")
    
    print(f"\nTo run inference on the trained model:")
    print(f"python examples/inference.py --model {final_model_path} --image <path_to_image>")
    
    print("\nNext steps:")
    print("- Replace sample data with real deepfake datasets")
    print("- Experiment with different model architectures")
    print("- Try different federated learning strategies")
    print("- Implement privacy-preserving techniques")


if __name__ == "__main__":
    demo_federated_learning()