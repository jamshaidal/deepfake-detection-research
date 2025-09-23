# Federated Learning for Deepfake Detection

A comprehensive implementation of federated learning for deepfake detection using PyTorch. This project enables distributed training of deepfake detection models across multiple clients while preserving data privacy.

## 🚀 Features

- **Federated Learning Framework**: Complete implementation of FedAvg algorithm
- **CNN-based Deepfake Detection**: State-of-the-art convolutional neural networks for deepfake detection
- **Face Extraction**: Automatic face detection and extraction from images/videos
- **Data Privacy**: Train models without centralizing sensitive data
- **Configurable Architecture**: Easy-to-modify configuration system
- **Comprehensive Evaluation**: Built-in metrics and visualization tools
- **Production Ready**: Modular design suitable for real-world deployment

## 📁 Project Structure

```
deepfake-detection-research/
├── src/
│   ├── models/
│   │   └── deepfake_cnn.py          # CNN models for deepfake detection
│   ├── data/
│   │   └── preprocessing.py         # Data loading and preprocessing
│   ├── federated/
│   │   ├── client.py               # Federated learning client
│   │   └── server.py               # Federated learning server
│   └── utils/
│       └── helpers.py              # Utility functions
├── examples/
│   ├── demo.py                     # Complete demo workflow
│   └── inference.py                # Inference script
├── tests/                          # Unit tests
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── train_federated.py             # Main training script
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- At least 8GB RAM

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/jamshaidal/deepfake-detection-research.git
cd deepfake-detection-research
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install additional system dependencies (for face recognition):**
```bash
# Ubuntu/Debian
sudo apt-get install cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev

# macOS
brew install cmake openblas lapack
```

## 🚀 Quick Start

### 1. Run the Demo

The easiest way to get started is with the demo script:

```bash
python examples/demo.py
```

This will:
- Create sample data
- Set up federated learning with 5 clients
- Train the model for 5 rounds
- Generate visualizations
- Save the trained model

### 2. Full Training

For full federated training with custom configuration:

```bash
python train_federated.py --config config.yaml --create-sample-data --rounds 100
```

### 3. Inference

Use a trained model for deepfake detection:

```bash
python examples/inference.py --model ./models/final_global_model.pth --image path/to/image.jpg
```

## ⚙️ Configuration

The `config.yaml` file contains all configurable parameters:

```yaml
# Federated Learning Configuration
federated_learning:
  num_clients: 5          # Number of federated clients
  rounds: 100             # Training rounds
  client_epochs: 5        # Local epochs per client
  batch_size: 32          # Batch size
  learning_rate: 0.001    # Learning rate

# Model Configuration
model:
  name: "DeepfakeDetectionCNN"  # Model architecture
  num_classes: 2               # Real vs Fake
  dropout_rate: 0.5           # Dropout for regularization

# Data Configuration
data:
  image_size: 224         # Input image size
  augmentation: true      # Enable data augmentation
  
# Training Configuration
training:
  device: "cuda"          # Device (cuda/cpu)
  save_model_path: "./models"
```

## 📊 Model Architectures

### DeepfakeDetectionCNN

A lightweight CNN optimized for federated learning:
- 4 convolutional blocks with batch normalization
- Adaptive pooling for variable input sizes
- Fully connected layers with dropout
- ~2.8M parameters

### ImprovedDeepfakeDetectionCNN

An advanced model with attention mechanism:
- Residual connections
- Spatial attention mechanism
- Global average pooling
- Enhanced feature extraction

## 🔬 Federated Learning Details

### Algorithm: FedAvg

The implementation uses the Federated Averaging (FedAvg) algorithm:

1. **Server** broadcasts global model to selected clients
2. **Clients** train locally on their private data
3. **Clients** send model updates back to server
4. **Server** aggregates updates using weighted averaging
5. **Repeat** for specified number of rounds

### Privacy Features

- **Data Locality**: Training data never leaves client devices
- **Model Updates Only**: Only model parameters are shared
- **Secure Aggregation**: Weighted averaging based on local data size

## 📈 Performance Monitoring

The framework includes comprehensive monitoring:

- **Training Metrics**: Loss, accuracy, training time per client
- **Global Metrics**: Aggregated performance across all clients
- **Visualizations**: Training curves, client participation, confusion matrices
- **Model Evaluation**: ROC curves, classification reports

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

For testing with sample data:

```bash
python train_federated.py --create-sample-data --rounds 5
```

## 📚 Usage Examples

### Custom Data

To use your own deepfake dataset:

1. **Organize data structure:**
```
data/
├── client_0/
│   ├── train/
│   │   ├── real/     # Real images
│   │   └── fake/     # Deepfake images
│   └── val/
│       ├── real/
│       └── fake/
├── client_1/
│   └── ...
```

2. **Update configuration:**
```yaml
data:
  data_dir: "./your_data_directory"
  extract_faces: true  # Enable face extraction
```

3. **Run training:**
```bash
python train_federated.py --data-dir ./your_data_directory
```

### Video Processing

For video datasets, extract frames first:

```python
from src.data.preprocessing import video_to_frames

frame_paths = video_to_frames(
    video_path="video.mp4",
    output_dir="./frames",
    max_frames=100
)
```

### Model Customization

Create custom models by extending the base CNN:

```python
from src.models.deepfake_cnn import DeepfakeDetectionCNN

class CustomModel(DeepfakeDetectionCNN):
    def __init__(self, num_classes=2):
        super().__init__(num_classes)
        # Add custom layers
```

## 🔧 Advanced Configuration

### Client Selection Strategies

Implement custom client selection:

```python
def custom_client_selection(available_clients, round_num):
    # Custom logic for client selection
    return selected_clients
```

### Data Distribution

For non-IID data distribution:

```python
from src.utils.helpers import prepare_federated_data

client_dirs = prepare_federated_data(
    base_data_dir="./data",
    num_clients=10,
    distribution="non_iid"  # or "iid"
)
```

### Hyperparameter Tuning

Key hyperparameters to tune:

- **Learning Rate**: Start with 0.001, adjust based on convergence
- **Batch Size**: Balance between speed and memory usage
- **Client Epochs**: More epochs = better local training, slower rounds
- **Number of Clients**: More clients = better privacy, slower training

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in `config.yaml`
   - Use smaller model architecture
   - Enable gradient checkpointing

2. **Face Detection Fails**:
   - Install dlib with CUDA support
   - Reduce image resolution
   - Disable face extraction temporarily

3. **Slow Training**:
   - Use GPU acceleration
   - Increase batch size
   - Reduce number of clients per round

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed
- **Data Loading**: Increase `num_workers` in data loaders
- **Memory**: Use `pin_memory=True` for faster GPU transfer

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest`
5. Submit pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{federated_deepfake_detection,
  title={Federated Learning for Deepfake Detection},
  author={Jamsha Idal},
  year={2024},
  url={https://github.com/jamshaidal/deepfake-detection-research}
}
```

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Face Recognition library for face detection
- OpenCV community for computer vision tools
- Research community working on federated learning and deepfake detection

## 📞 Support

For questions and support:

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/jamshaidal/deepfake-detection-research/issues)
- 📖 Documentation: [Project Wiki](https://github.com/jamshaidal/deepfake-detection-research/wiki)

---

**Note**: This implementation is for research and educational purposes. For production deployment, additional security measures and optimizations may be required.