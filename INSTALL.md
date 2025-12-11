# Installation Guide for Ctrl-World

## Installation Methods

### Method 1: Install in Development Mode (Recommended for development)

This allows you to edit the code and see changes immediately without reinstalling:

```bash
cd /data/Ctrl-World
pip install -e .
```

### Method 2: Install from Local Directory

```bash
pip install /data/Ctrl-World
```

### Method 3: Install from Git Repository

```bash
pip install git+https://github.com/riccardofeingold/Ctrl-World.git
```

### Method 4: Build and Install Distribution

```bash
cd /data/Ctrl-World
python setup.py sdist bdist_wheel
pip install dist/ctrl_world-0.1.0-py3-none-any.whl
```

## Usage After Installation

### Import as a Module

```python
from models.ctrl_world import CtrlWorldModel
from models.pipeline_ctrl_world import CtrlWorldPipeline
from dataset.dataset_droid_exp33 import DroidDataset
from config import wm_orca_args
```

### Use Command-Line Scripts

```bash
# Training
ctrl-world-train --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info

# Inference
ctrl-world-inference
```

## Uninstall

```bash
pip uninstall ctrl-world
```

## Notes

- Make sure to update the author information in `setup.py` and `pyproject.toml`
- The package excludes large directories like checkpoints, datasets, and logs
- Development mode (`-e`) is recommended if you're actively developing
