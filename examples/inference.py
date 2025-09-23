"""
Example script for running inference with the trained federated model
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.helpers import load_config, setup_logging
from models.deepfake_cnn import create_model
from data.preprocessing import get_transforms


def load_trained_model(model_path: str, config_path: str, device: torch.device):
    """
    Load a trained federated model
    
    Args:
        model_path: Path to the saved model
        config_path: Path to the configuration file
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create model
    model = create_model(config['model']).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"Model loaded from {model_path}")
    return model, config


def preprocess_image(image_path: str, config: dict):
    """
    Preprocess an image for inference
    
    Args:
        image_path: Path to the image
        config: Configuration dictionary
    
    Returns:
        Preprocessed image tensor
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config['data']['image_size'], config['data']['image_size']))
    image = Image.fromarray(image)
    
    # Apply transforms
    _, val_transform = get_transforms(config['data']['image_size'], augmentation=False)
    image_tensor = val_transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict_image(model: torch.nn.Module, 
                 image_tensor: torch.Tensor, 
                 device: torch.device):
    """
    Make prediction on an image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
    
    Returns:
        Prediction probabilities and class
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    return probabilities.cpu().numpy(), predicted_class.cpu().numpy()


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image for inference')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load trained model
        model, config = load_trained_model(args.model, args.config, device)
        
        # Preprocess image
        image_tensor = preprocess_image(args.image, config)
        
        # Make prediction
        probabilities, predicted_class = predict_image(model, image_tensor, device)
        
        # Interpret results
        class_names = ['Real', 'Fake']
        real_prob = probabilities[0][0] * 100
        fake_prob = probabilities[0][1] * 100
        prediction = class_names[predicted_class[0]]
        
        # Print results
        print("=" * 50)
        print("DEEPFAKE DETECTION RESULTS")
        print("=" * 50)
        print(f"Image: {args.image}")
        print(f"Prediction: {prediction}")
        print(f"Confidence:")
        print(f"  Real: {real_prob:.2f}%")
        print(f"  Fake: {fake_prob:.2f}%")
        print("=" * 50)
        
        # Save results
        results = {
            'image_path': args.image,
            'prediction': prediction,
            'real_probability': float(real_prob),
            'fake_probability': float(fake_prob),
            'predicted_class': int(predicted_class[0])
        }
        
        import json
        results_path = os.path.join(args.output_dir, 'inference_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()