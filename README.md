# isaac-sim-python: Python wrapper for NVIDIA Omniverse Isaac-Sim

## Overview
This repository contains a collection of python wrappers for NVIDIA Omniverse Isaac-Sim simulations. `grasp` package simulates a planar grasp execution of a Panda arm in a scene with various rigid objects place in a bin.

## Installation
This repository requires installation of NVIDIA Omniverse Isaac-Sim. A comprehensive setup tutorial is provided in the official [NVIDIA Omniverse Isaac-Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/setup.html) documentation. Following installation of Isaac-Sim, a conda environment must also be created that contains all the required packages for the python wrappers. Another comprehensive conda environment setup tutorial is provided in this [link](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/python_samples.html).

`ffmpeg-python` must be installed within the `isaac-sim` conda environment and can be aquired via a typical pip install:

```
conda activate isaac-sim
pip install ffmpeg-python
```

Lastly, clone the repository into the `python-samples` sub-directory within the `isaac-sim` directory.

```
git clone https://github.com/erasromani/isaac-sim-python.git
```

## Quickstart

Navigate to the `python-samples` sub-directory within the `isaac-sim` directory, source environment variables, activate conda environment, and run `simulate_grasp.py`.

```
source setenv.sh
conda activate isaac-sim
cd isaac-sim-python
python simulate_grasp.py -P Isaac/Props/Flip_Stack/large_corner_bracket_physics.usd Isaac/Props/Flip_Stack/screw_95_physics.usd Isaac/Props/Flip_Stack/t_connector_physics.usd -l nucleus_server -p 40 0 5 -a 45 -n 5 -v sim.mp4
```

The code above will simulate grasp execution of Panda arm in a scene with a bin and objects 5 randomly selected objects selected from the collection of usd files given. The specified grasp pose is a planar grasp with grasp position `(40, 0, 5)` and angle `5` degrees. A video of the simulation will be generated and saved as `sim.mp4`.

## Additional Resources
- https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
- https://docs.omniverse.nvidia.com/py/isaacsim/index.html
