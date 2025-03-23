# Biped locomotionmotion: Deep Phase Motion Generation Framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0) ![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg) ![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Biped locomotionmotion is a robot-agnostic framework that adapts the [DeepPhase](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2022) motion generation system ([Starke et al., 2022](https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2022/Paper.pdf)) for practical robotics applications. This implementation transforms the original animation-focused approach into a flexible framework for generating fluid, natural motions for various robot platforms.

Key contributions of this repo include:

* Pure Python-based data processing pipeline (removing Unity dependency)
* Robot-agnostic architecture supporting various robot configurations
* Streamlined data processing and training workflow
* Flexible deployment options for robotics applications

## Technical Architecture

The framework consists of two main components:

### 1. Periodic Autoencoder (PAE)

The PAE learns to encode motion data into a compact, periodic representation:

* **Input** : Joint states (positions and velocities) with window-based normalization
* **Output** : Periodic phase parameters (shift, frequency, amplitude)

### 2. Mode-Adaptive Neural Network (MANN)

The MANN uses the phase representation to predict motion:

* **Input** : Command velocities, robot state, and phase space representation
* **Output** : Next robot state, joint positions/velocities, contact states, and phase updates

## Installation

### Prerequisites

* Python 3.9
* CUDA-compatible GPU (recommended)
* Git LFS (for managing large data files)

### Clone the Repository

```bash
# Install Git LFS first if you haven't
apt-get install git-lfs  # Ubuntu/Debian
# OR
brew install git-lfs  # macOS

# Clone and initialize LFS
git clone https://github.com/ethanmclark1/biped_locomotion.git
cd biped_locomotion
git lfs install
git lfs pull
```

### Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate biped_locomotion

# Verify installation
python -c "import torch; import numpy as np; import quaternion; import scipy"
```

### Directory Structure

The repository is organized as follows:

```
biped_locomotion/
├── data/                # Motion capture data (LFS tracked)
├── scripts/             # Core implementation
│   ├── mann/            # Mode-Adaptive Neural Network
│   ├── pae/             # Periodic Autoencoder
│   ├── test/            # Test utilities
│   └── utils/           # Helper functions
├── ckpt/                # Model checkpoints (created during training)
└── img/                 # Output visualizations (created during training)
```

## Usage

> **Important** : All scripts should be run from the project root directory to avoid import errors!

### Data Preparation

Place your motion data in the `biped_locomotion/data/sequence_X/` directories with:

* `walking_joint_states.npy`: Joint angles in degrees
* `walking_root_states.npy`: Root position, orientation and velocities
* `walking_foot_contacts.npy`: Contact states

### Training Pipeline

#### 1. Train the Periodic Autoencoder (PAE)

```bash
# Run from project root
python -m biped_locomotion.scripts.pae.trainer
```

This learns to encode joint motions into periodic phase parameters.

#### 2. Train the Mode-Adaptive Neural Network (MANN)

```bash
# Run from project root 
python -m biped_locomotion.scripts.mann.trainer
```

This learns to predict motion based on command inputs and phase representation.

### Inference

To run inference and generate motion:

```bash
python -m biped_locomotion.scripts.mann.inference
```

### Configuration

Modify `biped_locomotion/scripts/config.yaml` to adjust:

* Network architecture parameters
* Training hyperparameters
* Data processing settings

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError: No module named 'biped_locomotion'`, ensure you're running scripts from the project root:

```bash
# Correct way to run scripts
cd /path/to/biped_locomotion
python -m biped_locomotion.scripts.pae.trainer
```

### CUDA Issues

If you encounter CUDA errors, verify your PyTorch installation matches your CUDA version:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

## Model Configuration

### PAE Configuration

* `phase_channels`: Number of phase channels (default: 8)
* `intermediate_channels`: Dimension of latent space (default: 40)
* `full_joint_state`: Whether to use both positions and velocities (default: False)

### MANN Configuration

* `gating_hidden`: Hidden layer size for gating network
* `main_hidden`: Hidden layer size for main network
* `n_experts`: Number of expert networks in mixture

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache 2.0 License - see [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite the original DeepPhase paper:

```bibtex
@inproceedings{starke2022deepphase,
  title={DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds},
  author={Starke, Sebastian and Mason, Ian and Komura, Taku},
  booktitle={ACM SIGGRAPH 2022 Conference Proceedings},
  year={2022},
  organization={ACM}
}
```
