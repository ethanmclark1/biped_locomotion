# Kinematic Predictor: A Robot-Agnostic Framework for Motion Generation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/licenses/MIT) ![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg) ![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Kinematic Predictor is a robot-agnostic framework that adapts the [DeepPhase](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2022) motion generation system ([Starke et al., 2022](https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2022/Paper.pdf)) for practical robotics applications. This implementation transforms the original animation-focused approach into a flexible framework for generating fluid, natural motions for various robot platforms.

Key contributions of this repo include:

* Pure Python-based data processing pipeline (removing Unity dependency)
* Robot-agnostic architecture supporting various robot configurations
* Streamlined data processing and training workflow
* Flexible deployment options for robotics applications

## Technical Architecture

### Input Processing (PAE)

Input Format:

* Motion capture sequence identifier
* Joint velocities (transformed into the root space with a window-based mean subtracted from it)

### Output Processing (MANN)

Input Features:

* Command velocities (xy linear, yaw)
* Projected gravity vector
* Joint positions and velocities
* Phase space representation

Output Features:

* Root state updates (position and velocity)
* Joint positions and velocities
* Contact states
* Phase vector updates

For detailed insights into the underlying architecture and theoretical foundations, please refer to the original [DeepPhase paper](https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2022/Paper.pdf).

## Setup

### Requirements

1. Create the conda environment

   ```
   conda env create -f environment.yml
   ```
2. Activate the environment

   ```
   conda activate kinematic_predictor
   ```
3. Verify the installation

   ```
   python -c "import torch; import numpy as np; import pandas as pd; import quaternion; import scipy"
   ```

### Optional Setup

1. Set up pre-commit hooks for development
   ```
   pre-commit install
   ```
2. Configure Weights & Biases for experiment tracking
   ```
   wandb login
   ```

## Robot Configuration

The framework is designed to work particularly with bipedal robots but can be adapted to quadrupeds as well. To configure for your platform:

1. Prepare your robot's state data:
   * Root state information (position, orientation, velocities)
   * Joint states (positions, velocities)
   * Contact information
2. Format your data according to the input/output specifications detailed in the documentation
3. Configure the network parameters based on your robot's degrees of freedom

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite the original DeepPhase paper:

<pre>

@inproceedings{starke2022deepphase,
  title={DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds},
  author={Starke, Sebastian and Mason, Ian and Komura, Taku},
  booktitle={ACM SIGGRAPH 2022 Conference Proceedings},
  year={2022},
  organization={ACM}
}
</pre>
