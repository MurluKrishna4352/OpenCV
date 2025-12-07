# 👐 OpenHands: Sign Language Recognition Library

> _Making Sign Language Recognition Accessible_

[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue)](https://openhands.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.7+-blue)](https://www.python.org/)

## Overview

OpenHands is a comprehensive Python library for sign language recognition powered by pose-based pretrained models. It provides tools for recognizing sign language across multiple languages with support for various datasets and neural network architectures.

The library features:
- **Multilingual Support**: Pre-trained models across multiple sign languages
- **Multiple Architectures**: ST-GCN, Decoupled-GCN, Pose-LSTM, Pose-Transformer, and more
- **Self-Supervised Learning**: DPC-based pre-training for improved performance
- **Easy-to-Use APIs**: Simple inference and training interfaces
- **Comprehensive Documentation**: Detailed guides and examples

## Quick Start

### Installation

**Stable version (recommended):**
```bash
pip install --upgrade OpenHands
```

**Latest development version:**
```bash
pip install git+https://github.com/AI4Bharat/OpenHands
```

### Basic Usage

```python
from openhands.apis import ClassificationModel

# Load a pre-trained model
model = ClassificationModel(config_path='path/to/config.yaml')

# Perform inference
predictions = model.predict('path/to/video.mp4')
print(predictions)
```

## Documentation

Comprehensive documentation is available at **[openhands.readthedocs.io](https://openhands.readthedocs.io)**, including:
- Installation guide
- API reference
- Training tutorials
- Dataset documentation
- Self-supervised learning guides

## Supported Datasets

The library supports multiple sign language datasets. Please cite the respective datasets in your research:

| Dataset | Language | Link |
| --- | --- | --- |
| AUTSL | Turkish | [Link](https://chalearnlap.cvc.uab.es/dataset/40/description/) |
| CSL | Chinese | [Link](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html) |
| DEVISIGN | Chinese | [Link](http://vipl.ict.ac.cn/homepage/ksl/data.html) |
| GSL | Greek | [Link](https://vcl.iti.gr/dataset/gsl/) |
| INCLUDE | Indian | [Link](https://sign-language.ai4bharat.org/#/INCLUDE) |
| LSA64 | Argentine | [Link](http://facundoq.github.io/datasets/lsa64/) |
| WLASL | English | [Link](https://dxli94.github.io/WLASL/) |

## Features

### Pose Extraction

Extract pose keypoints from videos using MediaPipe:
```bash
python scripts/mediapipe_extract.py --input video.mp4 --output poses.pkl
```

Or use the provided [extraction script](scripts/mediapipe_extract.py) for batch processing.

### Model Architectures

Supported neural network architectures:
- **ST-GCN**: Spatial-Temporal Graph Convolutional Networks
- **Decoupled-GCN**: Improved graph convolution
- **Pose-LSTM**: Recurrent neural networks for temporal modeling
- **Pose-Transformer**: Transformer-based architecture
- **SGN**: Skeleton-based Graph Convolution
- **GCN-BERT**: Graph convolution with BERT
- **DPC**: Deep Predictive Coding for self-supervised learning

### Training & Fine-tuning

Train models on your own datasets:
```bash
python examples/run_classifier.py --config path/to/config.yaml
```

## Project Structure

```
OpenHands/
├── openhands/                 # Main library code
│   ├── apis/                 # High-level APIs
│   ├── core/                 # Core utilities
│   ├── datasets/             # Dataset handling
│   ├── models/               # Neural network models
│   └── __init__.py
├── examples/                  # Example scripts
│   ├── configs/              # Configuration files for various datasets
│   ├── infer.py             # Inference example
│   ├── run_classifier.py    # Training example
│   └── run_dpc.py           # Self-supervised learning example
├── scripts/                   # Utility scripts
│   ├── mediapipe_extract.py # Pose extraction
│   └── ms_asl_downloader.py # Dataset utilities
├── docs/                      # Documentation
├── setup.py                   # Installation script
└── README.md                  # This file
```

## Examples

Several example scripts are provided in the `examples/` directory:

- **`infer.py`**: Perform inference on sign language videos
- **`run_classifier.py`**: Train a sign language classifier
- **`run_dpc.py`**: Self-supervised pre-training with Deep Predictive Coding

See the [documentation](https://openhands.readthedocs.io) for detailed examples.

## License

This project is released under the [Apache 2.0 License](LICENSE.txt).

## Citation

If you find OpenHands useful in your research, please cite us:

```bibtex
@misc{2021_openhands_slr_preprint,
  title={OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages},
  author={Prem Selvaraj and Gokul NC and Pratyush Kumar and Mitesh Khapra},
  year={2021},
  eprint={2110.05877},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@inproceedings{nc2022addressing,
  title={Addressing Resource Scarcity across Sign Languages with Multilingual Pretraining and Unified-Vocabulary Datasets},
  author={Gokul NC and Manideep Ladi and Sumit Negi and Prem Selvaraj and Pratyush Kumar and Mitesh M Khapra},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=zBBmV-i84Go}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests to help improve the library.

## Contact & Support

- **Documentation**: [openhands.readthedocs.io](https://openhands.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/AI4Bharat/OpenHands/issues)
- **Website**: [openhands.ai4bharat.org](https://openhands.ai4bharat.org)

---

**Made with ❤️ by [AI4Bhārat](https://ai4bharat.org)**
