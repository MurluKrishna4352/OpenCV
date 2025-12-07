# OpenHands: Sign Language Recognition Framework

A comprehensive Python framework for **sign language recognition**, **pose-based video understanding**, and **skeletal analysis** using state-of-the-art deep learning models.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Components](#core-components)
6. [Datasets](#datasets)
7. [Training Models](#training-models)
8. [Inference & Prediction](#inference--prediction)
9. [API Reference](#api-reference)
10. [Configuration Guide](#configuration-guide)
11. [Self-Supervised Learning](#self-supervised-learning)
12. [Visualization Tools](#visualization-tools)
13. [Scripts & Utilities](#scripts--utilities)
14. [Citation](#citation)

---

## 🎯 Project Overview

**OpenHands** is a research-grade framework designed to:
- Recognize and classify sign language gestures from videos
- Extract and analyze skeletal pose data using MediaPipe
- Train deep neural networks (GCN, STGCN, 3D-CNN, etc.)
- Deploy real-time inference systems for sign language translation
- Support self-supervised learning on unlabeled video data
- Handle 10+ sign language datasets globally

### Key Features
✅ Multi-dataset support (WLASL, MS-ASL, AUTSL, GSL, CSL, LSA64, etc.)  
✅ Modular architecture for easy customization  
✅ Pre-trained checkpoints available  
✅ Real-time inference pipeline  
✅ Self-supervised pretraining capabilities  
✅ Comprehensive documentation  
✅ MIT Licensed  

---

## 📁 Directory Structure

```
OpenHands/
├── openhands/                  # Main package
│   ├── apis/                   # High-level APIs
│   │   ├── classification_model.py   # Model inference
│   │   ├── dpc.py              # Dense Pose Correspondence
│   │   └── inference.py        # Real-time inference
│   ├── core/                   # Core utilities
│   │   ├── data.py             # Data loading & preprocessing
│   │   ├── exp_utils.py        # Experiment management
│   │   └── losses.py           # Custom loss functions
│   ├── datasets/               # Dataset handling
│   │   ├── data_readers.py     # Multi-dataset readers
│   │   ├── pose_transforms.py  # Pose augmentation
│   │   ├── video_transforms.py # Video preprocessing
│   │   ├── pipelines/          # Dataset-specific pipelines
│   │   ├── isolated/           # Isolated sign handling
│   │   ├── ssl/                # Self-supervised datasets
│   │   └── assets/             # Dataset assets
│   └── models/                 # Neural network models
│       ├── encoder/            # Feature extraction
│       ├── decoder/            # Output generation
│       ├── detection/          # Pose detection models
│       ├── ssl/                # SSL models
│       ├── loader.py           # Model loading utilities
│       └── network.py          # Main architectures
├── examples/                   # Example scripts
│   ├── run_classifier.py       # Training script
│   ├── run_dpc.py              # DPC training
│   ├── infer.py                # Inference script
│   └── configs/                # Configuration files
│       ├── wlasl/              # WLASL configs
│       ├── msasl/              # MS-ASL configs
│       ├── autsl/              # AUTSL configs
│       ├── asllvd/             # ASLLVD configs
│       ├── gsl/                # Greek Sign Language
│       ├── csl/                # Chinese Sign Language
│       ├── lsa64/              # Argentine Sign Language
│       ├── bosphorus22k/       # Turkish Sign Language
│       ├── devisign/           # DeviSign dataset
│       ├── fingerspelling/     # Fingerspelling dataset
│       ├── rwth-phoenix-weather-signer03-cutout/  # German SL
│       ├── multilingual/       # Multi-language configs
│       ├── include/            # Shared configs
│       └── ssl/                # Self-supervised configs
├── scripts/                    # Utility scripts
│   ├── mediapipe_extract.py    # Pose extraction
│   ├── ms_asl_downloader.py    # Dataset downloader
│   └── pkl_to_h5.py            # Format conversion
├── docs/                       # Sphinx documentation
│   ├── index.rst               # Documentation index
│   ├── api.rst                 # API documentation
│   ├── conf.py                 # Sphinx configuration
│   └── instructions/           # Detailed guides
│       ├── installation.rst    # Setup instructions
│       ├── training.rst        # Training guide
│       ├── inference.rst       # Inference guide
│       ├── models.rst          # Model descriptions
│       ├── datasets.rst        # Dataset guide
│       ├── features.rst        # Feature extraction
│       ├── self_supervised.rst # SSL guide
│       └── support.rst         # Support resources
├── AUTSL/                      # AUTSL dataset (sample)
│   ├── holistic_poses/         # Extracted poses
│   └── train/                  # Training splits
├── dummy_videos/               # Sample videos for testing
├── output/                     # Output directory
├── .gitignore                  # Git ignore rules
├── .readthedocs.yml            # Read the Docs config
├── setup.py                    # Package setup
├── setup.cfg                   # Setup configuration
├── LICENSE.txt                 # MIT License
├── CITATION.cff                # Citation format
├── extract_single.py           # Single video processing
├── visualize_keypoints.py      # Keypoint visualization
├── visualize_output.py         # Output visualization
├── apple.csv                   # Sample data
└── README.md                   # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA 11.0+ (for GPU support, optional)

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd OpenHands

# Install in development mode
pip install -e .

# Or standard installation
pip install .
```

### From Requirements

```bash
pip install -r docs/requirements.txt
```

### Key Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
mediapipe>=0.8.0
numpy
pandas
opencv-python
pyyaml
scikit-learn
tensorboard
```

---

## 🚀 Quick Start

### 1. Extract Poses from Video

```python
from openhands.apis import inference

# Extract skeletal poses from a video
poses = inference.extract_poses("path/to/video.mp4")
print(poses.shape)  # (frames, keypoints, 3)
```

### 2. Make Predictions

```python
from openhands.apis import classification_model

# Load model and make predictions
model = classification_model.ClassificationModel(
    config_path="examples/configs/wlasl/decoupled_gcn.yaml",
    checkpoint_path="path/to/checkpoint.pt"
)

predictions = model.predict("path/to/video.mp4")
print(predictions)  # {class_id: probability, ...}
```

### 3. Train a Model

```bash
# From command line
python examples/run_classifier.py \
    --config examples/configs/wlasl/decoupled_gcn.yaml \
    --dataset wlasl \
    --batch-size 32 \
    --epochs 50
```

---

## 🧠 Core Components

### 1. **APIs** (`openhands/apis/`)

#### `classification_model.py`
- **Purpose**: High-level interface for classification tasks
- **Key Classes**:
  - `ClassificationModel` - Main inference class
- **Usage**:
  ```python
  from openhands.apis.classification_model import ClassificationModel
  
  model = ClassificationModel(config_path, checkpoint_path)
  predictions = model.predict(video_path)
  ```

#### `dpc.py` (Dense Pose Correspondence)
- **Purpose**: Self-supervised learning using pose correspondence
- **Key Functions**:
  - `DPCLoss` - Loss function for DPC training
  - `DPCModel` - Model wrapper
- **Usage**:
  ```python
  from openhands.apis.dpc import DPCModel
  
  dpc_model = DPCModel(config)
  loss = dpc_model.forward(video_frames)
  ```

#### `inference.py`
- **Purpose**: Real-time inference pipeline
- **Key Functions**:
  - `extract_poses()` - Extract skeletal poses
  - `predict_from_video()` - End-to-end prediction
  - `get_keypoints()` - Get pose keypoints
- **Usage**:
  ```python
  from openhands.apis.inference import extract_poses
  
  poses = extract_poses("video.mp4", model_type="mediapipe")
  ```

### 2. **Core Utilities** (`openhands/core/`)

#### `data.py`
- **Purpose**: Data loading, preprocessing, and batching
- **Key Classes**:
  - `DataLoader` - Custom data loader
  - `DataProcessor` - Preprocessing pipeline
- **Functionality**:
  - Load pose sequences
  - Normalize skeletal data
  - Handle missing keypoints
  - Create train/val/test splits

#### `exp_utils.py`
- **Purpose**: Experiment management and logging
- **Key Functions**:
  - `setup_experiment()` - Initialize experiment
  - `save_checkpoint()` - Save model states
  - `load_checkpoint()` - Resume training
  - `log_metrics()` - Track performance
- **Usage**:
  ```python
  from openhands.core.exp_utils import setup_experiment
  
  exp_dir = setup_experiment("my_experiment")
  ```

#### `losses.py`
- **Purpose**: Custom loss functions
- **Available Losses**:
  - `SequenceLoss` - Sequential data loss
  - `ContrastiveLoss` - For self-supervised learning
  - `FocalLoss` - For imbalanced datasets
  - `TemporalLoss` - Temporal coherence loss
- **Usage**:
  ```python
  from openhands.core.losses import SequenceLoss
  
  loss_fn = SequenceLoss()
  loss = loss_fn(predictions, targets)
  ```

### 3. **Datasets** (`openhands/datasets/`)

#### `data_readers.py`
- **Purpose**: Unified interface for multiple datasets
- **Supported Datasets**:
  - WLASL (World's Largest Sign Language)
  - MS-ASL (Microsoft American Sign Language)
  - AUTSL (Autistic Sign Language)
  - GSL (Greek Sign Language)
  - CSL (Chinese Sign Language)
  - LSA64 (Argentine Sign Language)
  - ASLLVD (ASL Linguistic Video)
  - Bosphorus22k (Turkish Sign Language)
  - DeviSign, Fingerspelling, RWTH-PHOENIX, Multilingual
- **Key Classes**:
  - `DatasetReader` - Base class
  - `WLASLReader`, `MSASLReader`, etc. - Dataset-specific readers
- **Usage**:
  ```python
  from openhands.datasets.data_readers import WLASLReader
  
  dataset = WLASLReader(split="train")
  videos, labels = dataset.load_data()
  ```

#### `pose_transforms.py`
- **Purpose**: Pose-specific data augmentation
- **Available Transforms**:
  - `RotatePose` - Rotate skeleton
  - `ScalePose` - Scale skeleton
  - `FlipPose` - Mirror pose
  - `NoisePose` - Add noise to joints
  - `TemporalCrop` - Temporal window cropping
  - `SpeedJitter` - Speed variation
- **Usage**:
  ```python
  from openhands.datasets.pose_transforms import RotatePose, ScalePose
  
  transforms = [RotatePose(angle=15), ScalePose(scale=0.1)]
  augmented_pose = transforms[0](pose)
  ```

#### `video_transforms.py`
- **Purpose**: Video preprocessing and augmentation
- **Available Transforms**:
  - `ResizeVideo` - Resize frames
  - `CenterCrop` - Crop frames
  - `RandomCrop` - Random cropping
  - `Normalize` - Normalize pixel values
  - `ToTensor` - Convert to tensor
  - `RandomFlip` - Horizontal flip
- **Usage**:
  ```python
  from openhands.datasets.video_transforms import ResizeVideo, Normalize
  
  transforms = [ResizeVideo(224), Normalize()]
  ```

#### `pipelines/` - Dataset Pipelines
- **Purpose**: Complete data processing pipelines
- **Structure**:
  - Each dataset has a dedicated pipeline file
  - Handles dataset-specific preprocessing
  - Manages splits and annotations
- **Example**:
  ```python
  from openhands.datasets.pipelines.wlasl import WLASLPipeline
  
  pipeline = WLASLPipeline(config)
  train_loader = pipeline.get_train_loader()
  ```

#### `ssl/` - Self-Supervised Learning Datasets
- **Purpose**: Datasets for unsupervised pretraining
- **Available Methods**:
  - Dense Pose Correspondence (DPC)
  - Contrastive learning
  - Temporal coherence
- **Usage**:
  ```python
  from openhands.datasets.ssl import SSLDataset
  
  ssl_dataset = SSLDataset(video_dir="path/to/videos")
  ```

### 4. **Models** (`openhands/models/`)

#### `network.py`
- **Purpose**: Main neural network architectures
- **Available Models**:
  - **Graph Convolutional Networks (GCN)**
    - Decoupled GCN
    - Spatial-Temporal GCN (STGCN)
    - Dynamic GCN
  - **Recurrent Networks**
    - LSTM-based models
    - GRU-based models
  - **Convolutional Networks**
    - 3D-CNN
    - 2D-CNN + RNN hybrids
  - **Transformer-based**
    - Vision Transformer
    - BERT-style architectures
- **Usage**:
  ```python
  from openhands.models.network import DecoupledGCN
  
  model = DecoupledGCN(
      num_classes=1000,
      in_channels=3,
      hidden_channels=64
  )
  ```

#### `encoder/` - Feature Encoders
- **Purpose**: Extract features from skeletal poses
- **Available Encoders**:
  - `PoseEncoder` - Pose feature extraction
  - `TemporalEncoder` - Temporal modeling
  - `SpatialEncoder` - Spatial structure encoding
- **Usage**:
  ```python
  from openhands.models.encoder import PoseEncoder
  
  encoder = PoseEncoder(input_dim=3, output_dim=256)
  features = encoder(pose_sequence)
  ```

#### `decoder/` - Output Decoders
- **Purpose**: Generate predictions from features
- **Available Decoders**:
  - `ClassificationDecoder` - For classification
  - `RegressionDecoder` - For regression tasks
  - `SequenceDecoder` - For sequence generation
- **Usage**:
  ```python
  from openhands.models.decoder import ClassificationDecoder
  
  decoder = ClassificationDecoder(input_dim=256, num_classes=1000)
  logits = decoder(features)
  ```

#### `detection/` - Pose Detection
- **Purpose**: Detect poses in raw video frames
- **Available Models**:
  - MediaPipe Holistic
  - OpenPose
  - YOLOv5-Pose
- **Usage**:
  ```python
  from openhands.models.detection import MediaPipeDetector
  
  detector = MediaPipeDetector()
  poses = detector.detect(frame)
  ```

#### `ssl/` - Self-Supervised Models
- **Purpose**: Models for unsupervised pretraining
- **Available Methods**:
  - Dense Pose Correspondence
  - Temporal cluster matching
  - Contrastive learning frameworks
- **Usage**:
  ```python
  from openhands.models.ssl import DPCNet
  
  model = DPCNet(config)
  ```

#### `loader.py`
- **Purpose**: Model loading and management
- **Key Functions**:
  - `load_model()` - Load pretrained models
  - `get_model()` - Initialize models
  - `save_model()` - Save model weights
  - `list_available_models()` - Show available models
- **Usage**:
  ```python
  from openhands.models.loader import load_model
  
  model = load_model("wlasl", "decoupled_gcn", checkpoint_path)
  ```

---

## 📊 Datasets

### Supported Sign Language Datasets

| Dataset | Acronym | Signs | Videos | Link |
|---------|---------|-------|--------|------|
| World's Largest Sign Language | WLASL | 1000+ | 21,083 | config: `wlasl/` |
| Microsoft American Sign Language | MS-ASL | 1000 | 16,000 | config: `msasl/` |
| Autistic Sign Language | AUTSL | 100 | 3,000+ | config: `autsl/` |
| Greek Sign Language | GSL | 320 | 10,000+ | config: `gsl/` |
| Chinese Sign Language | CSL | 500+ | 5,000+ | config: `csl/` |
| Argentine Sign Language | LSA64 | 64 | 3,200 | config: `lsa64/` |
| ASL Linguistic Video | ASLLVD | 94 | 9,900 | config: `asllvd/` |
| Turkish Sign Language | Bosphorus22k | 417 | 22,000 | config: `bosphorus22k/` |
| DeviSign | DeviSign | 34 | 1,078 | config: `devisign/` |
| Fingerspelling | Fingerspelling | Various | Custom | config: `fingerspelling/` |
| German SL (RWTH) | RWTH-PHOENIX | 1200+ | 7,000+ | config: `rwth-phoenix-*/` |

### Loading a Dataset

```python
from openhands.datasets.data_readers import WLASLReader

# Initialize dataset
dataset = WLASLReader(
    split="train",
    pose_format="mediapipe",
    normalize=True
)

# Get data
for video, label in dataset:
    print(video.shape)  # (frames, keypoints, 3)
    print(label)
```

### Dataset Configuration Files

Location: `examples/configs/{dataset_name}/`

Each dataset has YAML config files defining:
- Data paths
- Train/val/test splits
- Preprocessing parameters
- Model hyperparameters
- Augmentation strategies

---

## 🏋️ Training Models

### Training Script: `examples/run_classifier.py`

#### Basic Training

```bash
python examples/run_classifier.py \
    --config examples/configs/wlasl/decoupled_gcn.yaml \
    --dataset wlasl \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 0.001
```

#### Advanced Training Options

```bash
python examples/run_classifier.py \
    --config examples/configs/wlasl/decoupled_gcn.yaml \
    --dataset wlasl \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 0.001 \
    --optimizer adam \
    --scheduler cosine \
    --warmup-epochs 5 \
    --weight-decay 1e-4 \
    --dropout 0.5 \
    --data-augment \
    --use-gpu \
    --gpu-id 0 \
    --seed 42 \
    --save-interval 5 \
    --log-interval 100
```

#### Programmatic Training

```python
from examples.run_classifier import Trainer
import yaml

# Load config
with open("examples/configs/wlasl/decoupled_gcn.yaml") as f:
    config = yaml.safe_load(f)

# Initialize trainer
trainer = Trainer(config, dataset_name="wlasl")

# Train model
best_model = trainer.train(
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)

# Save model
trainer.save_checkpoint("output/checkpoint.pt")
```

### Configuration File Structure (YAML)

```yaml
# examples/configs/wlasl/decoupled_gcn.yaml
dataset:
  name: wlasl
  split_ratio: [0.7, 0.15, 0.15]  # train, val, test
  batch_size: 32
  num_workers: 4
  normalize: true
  augment: true

model:
  name: decoupled_gcn
  num_classes: 1000
  in_channels: 3  # x, y, confidence
  hidden_channels: 64
  num_layers: 4
  dropout: 0.5

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  scheduler: cosine
  warmup_epochs: 5
  weight_decay: 1e-4
  loss: cross_entropy

augmentation:
  pose_rotation: 15  # degrees
  pose_scale: 0.1
  temporal_crop: true
  temporal_jitter: true
```

### Training Monitoring

```bash
# Monitor with TensorBoard
tensorboard --logdir output/runs
```

---

## 🔮 Inference & Prediction

### Quick Inference: `examples/infer.py`

```bash
python examples/infer.py \
    --config examples/configs/wlasl/decoupled_gcn.yaml \
    --checkpoint path/to/checkpoint.pt \
    --video path/to/video.mp4 \
    --output output/prediction.json
```

### Programmatic Inference

```python
from openhands.apis.classification_model import ClassificationModel

# Load model
model = ClassificationModel(
    config_path="examples/configs/wlasl/decoupled_gcn.yaml",
    checkpoint_path="path/to/checkpoint.pt",
    device="cuda"
)

# Make prediction
result = model.predict(
    video_path="path/to/video.mp4",
    return_top_k=5,
    confidence_threshold=0.5
)

print(result)
# Output:
# {
#     'predicted_class': 'hello',
#     'class_id': 42,
#     'confidence': 0.98,
#     'top_5': [
#         {'class': 'hello', 'id': 42, 'confidence': 0.98},
#         {'class': 'hi', 'id': 100, 'confidence': 0.01},
#         ...
#     ]
# }
```

### Single Video Processing: `extract_single.py`

```bash
python extract_single.py \
    --video path/to/video.mp4 \
    --output path/to/poses.npy \
    --model-type mediapipe \
    --save-format numpy
```

### Programmatic Extraction

```python
from openhands.apis.inference import extract_poses
import numpy as np

# Extract poses
poses = extract_poses(
    video_path="path/to/video.mp4",
    model_type="mediapipe",
    frame_rate=30
)

print(poses.shape)  # (frames, 33, 3) for mediapipe
# 33 keypoints: face (468), pose (33), hands (21*2)
# 3 values: x, y, confidence

# Save poses
np.save("poses.npy", poses)
```

---

## 📚 API Reference

### Classification Model API

```python
from openhands.apis.classification_model import ClassificationModel

class ClassificationModel:
    def __init__(self, config_path, checkpoint_path, device='cpu'):
        """Initialize model"""
        
    def predict(self, video_path, return_top_k=1, confidence_threshold=0.0):
        """Make prediction on video"""
        
    def predict_batch(self, video_list):
        """Batch prediction"""
        
    def get_model(self):
        """Get underlying model"""
        
    def to(self, device):
        """Move model to device"""
```

### Inference API

```python
from openhands.apis.inference import *

def extract_poses(video_path, model_type='mediapipe', **kwargs):
    """Extract poses from video"""
    
def predict_from_video(video_path, config_path, checkpoint_path):
    """End-to-end prediction"""
    
def get_keypoints(frame, model_type='mediapipe'):
    """Get keypoints from single frame"""
    
def draw_poses(frame, poses):
    """Draw poses on frame"""
```

### Data Loading API

```python
from openhands.datasets.data_readers import *

class DatasetReader:
    def __init__(self, split, **kwargs):
        """Initialize dataset"""
        
    def __len__(self):
        """Get dataset size"""
        
    def __getitem__(self, idx):
        """Get single sample"""
        
    def load_data(self):
        """Load all data"""
        
    def get_loader(self, batch_size, shuffle=False):
        """Get PyTorch DataLoader"""
```

### Model Loading API

```python
from openhands.models.loader import *

def load_model(dataset_name, model_name, checkpoint_path=None):
    """Load pretrained model"""
    
def get_model(config):
    """Initialize model from config"""
    
def save_model(model, save_path):
    """Save model weights"""
    
def list_available_models():
    """List available models"""
```

---

## ⚙️ Configuration Guide

### Config File Locations
```
examples/configs/
├── wlasl/
│   ├── decoupled_gcn.yaml
│   ├── stgcn.yaml
│   ├── 3d_cnn.yaml
│   └── ...
├── msasl/
├── autsl/
├── asllvd/
├── gsl/
├── csl/
├── lsa64/
├── bosphorus22k/
├── devisign/
├── fingerspelling/
├── rwth-phoenix-weather-signer03-cutout/
├── multilingual/
├── include/          # Shared configs
└── ssl/              # Self-supervised configs
```

### Config Parameters Explained

```yaml
# Data configuration
dataset:
  name: wlasl                    # Dataset name
  data_root: /path/to/data       # Root directory
  pose_format: mediapipe         # Pose format (mediapipe, openpose, etc.)
  normalize: true                # Normalize poses
  scale: [224, 224]              # Frame scale
  split_ratio: [0.7, 0.15, 0.15] # train/val/test split
  batch_size: 32                 # Batch size
  num_workers: 4                 # Data loading workers
  shuffle: true                  # Shuffle training data

# Model configuration
model:
  name: decoupled_gcn            # Model architecture
  num_classes: 1000              # Number of classes
  in_channels: 3                 # Input channels (x, y, conf)
  hidden_channels: 64            # Hidden dimensions
  num_layers: 4                  # Number of layers
  dropout: 0.5                   # Dropout rate
  activation: relu               # Activation function

# Training configuration
training:
  epochs: 100                    # Total epochs
  learning_rate: 0.001           # Learning rate
  optimizer: adam                # Optimizer (adam, sgd, etc.)
  scheduler: cosine              # LR scheduler
  warmup_epochs: 5               # Warmup epochs
  weight_decay: 1e-4             # L2 regularization
  loss: cross_entropy            # Loss function
  gradient_clip: 1.0             # Gradient clipping

# Augmentation configuration
augmentation:
  enabled: true                  # Enable augmentation
  pose_rotation: 15              # Max rotation (degrees)
  pose_scale: [0.9, 1.1]         # Scale range
  pose_flip: true                # Horizontal flip
  temporal_crop: true            # Temporal cropping
  temporal_jitter: 0.1           # Time jitter
  noise_sigma: 0.01              # Gaussian noise
```

---

## 🔒 Self-Supervised Learning

### Overview

Self-supervised learning allows pretraining on unlabeled video data, improving downstream performance.

### Available Methods

1. **Dense Pose Correspondence (DPC)**
   - Learns by matching poses across videos
   - Best for sign language datasets
   
2. **Temporal Coherence**
   - Exploits temporal structure
   - Unsupervised representation learning
   
3. **Contrastive Learning**
   - Learns discriminative features
   - Positive/negative pair matching

### Training SSL Models: `examples/run_dpc.py`

```bash
python examples/run_dpc.py \
    --config examples/configs/ssl/pretrain_dpc_decoupled_gcn.yaml \
    --data-root /path/to/unlabeled/videos \
    --batch-size 64 \
    --epochs 200 \
    --learning-rate 0.001
```

### Programmatic SSL Training

```python
from openhands.apis.dpc import DPCModel
from openhands.datasets.ssl import SSLDataset
import torch

# Load SSL config
import yaml
with open("examples/configs/ssl/pretrain_dpc_decoupled_gcn.yaml") as f:
    config = yaml.safe_load(f)

# Create dataset
ssl_dataset = SSLDataset(
    video_dir="/path/to/videos",
    config=config
)
loader = torch.utils.data.DataLoader(ssl_dataset, batch_size=64)

# Initialize model
dpc_model = DPCModel(config)

# Train
for epoch in range(200):
    for batch in loader:
        loss = dpc_model.train_step(batch)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Save pretrained model
dpc_model.save("output/pretrained.pt")
```

### Using Pretrained SSL Models

```python
from openhands.models.loader import load_model

# Load pretrained model
model = load_model(
    dataset_name="ssl",
    model_name="dpc_decoupled_gcn",
    checkpoint_path="output/pretrained.pt"
)

# Fine-tune on downstream task
from openhands.core.data import DataLoader
import torch.optim as optim

dataset = WLASLReader(split="train")
loader = DataLoader(dataset, batch_size=32)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(50):
    for batch in loader:
        predictions = model(batch)
        loss = ...  # compute loss
        loss.backward()
        optimizer.step()
```

---

## 🎨 Visualization Tools

### Keypoint Visualization: `visualize_keypoints.py`

```bash
python visualize_keypoints.py \
    --video path/to/video.mp4 \
    --output path/to/output.mp4 \
    --model-type mediapipe \
    --draw-connections true
```

### Programmatic Visualization

```python
from openhands.apis.inference import extract_poses, draw_poses
import cv2

# Load video
cap = cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Process and visualize
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract poses
    poses = extract_poses(frame)
    
    # Draw on frame
    annotated_frame = draw_poses(frame, poses)
    
    cv2.imshow("Poses", annotated_frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Output Visualization: `visualize_output.py`

```bash
python visualize_output.py \
    --video path/to/video.mp4 \
    --predictions output/prediction.json \
    --output path/to/annotated_video.mp4
```

---

## 🛠️ Scripts & Utilities

### 1. **Extract Poses from Videos**: `scripts/mediapipe_extract.py`

Extract skeletal poses from video files using MediaPipe.

```bash
python scripts/mediapipe_extract.py \
    --input-dir /path/to/videos \
    --output-dir /path/to/poses \
    --model-type holistic \
    --save-format npy
```

**Options:**
- `--input-dir`: Directory with videos
- `--output-dir`: Save directory for poses
- `--model-type`: holistic, pose, or hands
- `--save-format`: npy, pkl, or h5
- `--batch-size`: Processing batch size
- `--gpu`: Use GPU

### 2. **Download MS-ASL Dataset**: `scripts/ms_asl_downloader.py`

Automatically download and organize MS-ASL dataset.

```bash
python scripts/ms_asl_downloader.py \
    --output-dir /path/to/msasl \
    --num-workers 8 \
    --verify-ssl true
```

**Features:**
- Parallel downloading
- Automatic organization
- Resume capability
- Data validation

### 3. **Convert Data Formats**: `scripts/pkl_to_h5.py`

Convert pose data between formats (pickle, HDF5, NumPy).

```bash
python scripts/pkl_to_h5.py \
    --input-file data.pkl \
    --output-file data.h5 \
    --input-format pkl \
    --output-format h5
```

---

## 📖 Documentation

Full documentation available at `docs/`

**Build Documentation:**
```bash
cd docs
make html
# Output: docs/_build/html/index.html
```

**Documentation Sections:**
- `installation.rst` - Setup guide
- `training.rst` - Training pipeline
- `inference.rst` - Inference guide
- `models.rst` - Model descriptions
- `datasets.rst` - Dataset guide
- `features.rst` - Feature extraction
- `self_supervised.rst` - SSL guide
- `support.rst` - Support & troubleshooting

---

## 📦 Package Structure

### `openhands/__init__.py`
```python
from .apis import *
from .models import *
from .datasets import *
from .core import *
```

### `openhands/__version__.py`
Current version information.

### Key Import Paths

```python
# APIs
from openhands.apis import ClassificationModel, inference, dpc

# Models
from openhands.models import loader, network
from openhands.models.encoder import PoseEncoder
from openhands.models.decoder import ClassificationDecoder

# Datasets
from openhands.datasets import data_readers, pose_transforms, video_transforms
from openhands.datasets.pipelines.wlasl import WLASLPipeline

# Core utilities
from openhands.core import data, exp_utils, losses
```

---

## 🔧 Common Tasks

### Task 1: Train on Custom Sign Language Dataset

```python
from openhands.datasets.pipelines import create_custom_pipeline
from openhands.models.loader import get_model
import torch.optim as optim

# 1. Create custom dataset pipeline
pipeline = create_custom_pipeline(
    video_dir="/path/to/videos",
    annotation_file="annotations.json",
    num_classes=100
)

# 2. Load model
model = get_model({
    'name': 'decoupled_gcn',
    'num_classes': 100,
    'in_channels': 3
})

# 3. Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader = pipeline.get_train_loader(batch_size=32)

for epoch in range(100):
    for poses, labels in train_loader:
        pred = model(poses)
        loss = cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Task 2: Deploy Model for Real-Time Prediction

```python
from openhands.apis import ClassificationModel
import cv2

# Load model
model = ClassificationModel(
    config_path="config.yaml",
    checkpoint_path="model.pt",
    device="cuda"
)

# Real-time inference
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predict
    result = model.predict_frame(frame)
    
    # Display
    cv2.putText(frame, f"Sign: {result['class']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Task 3: Fine-tune Pretrained Model

```python
from openhands.models.loader import load_model
from openhands.datasets.data_readers import CustomReader
import torch.optim as optim

# Load pretrained model
model = load_model(
    dataset_name="wlasl",
    model_name="decoupled_gcn",
    checkpoint_path="pretrained.pt"
)

# Freeze backbone, train head
for param in model.backbone.parameters():
    param.requires_grad = False

# Load custom dataset
dataset = CustomReader(split="train")
loader = dataset.get_loader(batch_size=32)

# Fine-tune
optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                              model.parameters()), lr=0.0001)

for epoch in range(50):
    for poses, labels in loader:
        pred = model(poses)
        loss = ...
        loss.backward()
        optimizer.step()
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```bash
# Reduce batch size
python examples/run_classifier.py --batch-size 16

# Or enable gradient accumulation
python examples/run_classifier.py --grad-accumulate-steps 2
```

### Issue: Model Not Converging
**Solution:**
```yaml
# Adjust learning rate in config
training:
  learning_rate: 0.0001  # Reduce by 10x
  scheduler: cosine
  warmup_epochs: 10      # Increase warmup
```

### Issue: Poor Pose Detection
**Solution:**
```python
# Use high-resolution frames
poses = extract_poses(
    video_path,
    model_type='mediapipe',
    frame_scale=(1024, 1024)  # Increase from default
)
```

---

## 📄 Citation

If you use OpenHands in your research, please cite:

```bibtex
@software{openhands,
  title={OpenHands: Sign Language Recognition Framework},
  author={Your Authors},
  year={2024},
  url={https://github.com/...}
}
```

See `CITATION.cff` for alternative citation formats.

---

## 📜 License

This project is licensed under the MIT License - see `LICENSE.txt` for details.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Support & Resources

**Documentation**: `docs/` directory  
**Issues**: GitHub Issues  
**Discussions**: GitHub Discussions  

**Quick Links:**
- [Read the Docs](https://openhands.readthedocs.io)
- [WLASL Dataset](https://github.com/dxli94/WLASL)
- [MediaPipe](https://google.github.io/mediapipe/)

---

## 🗺️ Roadmap

- [ ] Add more sign language datasets
- [ ] Implement continuous sign recognition
- [ ] Multi-signer support
- [ ] Real-time translation
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Web interface

---

**Last Updated:** December 2, 2024  
**Version:** 1.0.0