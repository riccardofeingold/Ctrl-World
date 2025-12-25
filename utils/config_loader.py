"""
Configuration loading utilities with support for variable interpolation.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
from dataclasses import fields, asdict
from typing import Any, Dict, Optional
from omegaconf import OmegaConf, DictConfig
from config import wm_orca_args


def load_experiment_config(config_path: str, base_args: Optional[wm_orca_args] = None) -> wm_orca_args:
    """
    Load experiment configuration from YAML with variable interpolation support.
    Automatically detects and merges with base config if specified in the experiment config.
    
    Args:
        config_path: Path to experiment YAML config
        base_args: Base config dataclass (default: wm_orca_args())
    
    Returns:
        Updated config dataclass with interpolated values
    """
    if base_args is None:
        base_args = wm_orca_args()
    
    # Load YAML config with OmegaConf for interpolation support
    omega_config = OmegaConf.load(config_path)
    
    # Check if experiment specifies a base config
    if 'experiment' in omega_config and 'base_config' in omega_config.experiment:
        base_config_path = omega_config.experiment.base_config
        # Resolve relative path
        if not os.path.isabs(base_config_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            base_config_path = os.path.normpath(os.path.join(config_dir, base_config_path))
        
        if os.path.exists(base_config_path):
            print(f"📂 Loading base config: {base_config_path}")
            base_omega = OmegaConf.load(base_config_path)
            # Merge: base config first, then experiment config (experiment overrides)
            omega_config = OmegaConf.merge(base_omega, omega_config)
        else:
            print(f"⚠️  Base config not found: {base_config_path}")
    
    # Resolve all interpolations (${...} references)
    OmegaConf.resolve(omega_config)
    
    # Convert to plain dict
    exp_config = OmegaConf.to_container(omega_config, resolve=True)
    
    # Flatten nested config structure
    flat_config = _flatten_config(exp_config)
    
    # Update base_args with values from YAML
    for key, value in flat_config.items():
        if hasattr(base_args, key):
            setattr(base_args, key, value)
        else:
            print(f"Warning: Config key '{key}' not found in base config, skipping...")
    
    # Compute derived parameters
    _compute_derived_params(base_args)
    
    return base_args


def _flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary structure.
    
    Args:
        config: Nested configuration dictionary
        parent_key: Parent key for recursion
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and not _is_special_dict(k):
            items.extend(_flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)


def _is_special_dict(key: str) -> bool:
    """Check if a key represents a special dictionary that shouldn't be flattened."""
    special_keys = ['gripper_max_dict', 'model_paths']
    return key in special_keys


def _compute_derived_params(args: wm_orca_args):
    """Compute derived parameters based on config values."""
    # Compute down_sample from fps
    if hasattr(args, 'original_fps') and hasattr(args, 'fps'):
        args.down_sample = int(args.original_fps / args.fps)
    
    # Ensure dataset_cfgs matches dataset_names if not explicitly set
    if hasattr(args, 'dataset_names') and not hasattr(args, 'dataset_cfgs'):
        args.dataset_cfgs = args.dataset_names
    
    # Update output paths based on tag
    if hasattr(args, 'tag'):
        if not args.output_dir or args.output_dir == f"model_ckpt/{args.tag}":
            args.output_dir = f"model_ckpt/{args.tag}"
    
    # Convert dtype string to torch dtype
    if hasattr(args, 'dtype') and isinstance(args.dtype, str):
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        args.dtype = dtype_map.get(args.dtype.lower(), torch.bfloat16)


def save_config_to_yaml(args: wm_orca_args, save_path: str):
    """
    Save current config to YAML for reproducibility.
    
    Args:
        args: Configuration dataclass
        save_path: Path to save YAML file
    """
    config_dict = {}
    
    # Iterate through all instance attributes (including class variables)
    for attr_name in dir(args):
        # Skip private/magic methods and callables
        if attr_name.startswith('_') or callable(getattr(args, attr_name)):
            continue
        
        try:
            value = getattr(args, attr_name)
            
            # Convert torch dtypes to strings
            if isinstance(value, torch.dtype):
                dtype_str_map = {
                    torch.float32: 'float32',
                    torch.float16: 'float16',
                    torch.bfloat16: 'bfloat16',
                }
                value = dtype_str_map.get(value, 'bfloat16')
            
            config_dict[attr_name] = value
        except Exception:
            # Skip attributes that can't be accessed or serialized
            continue
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Config saved to {save_path}")


def list_available_experiments(experiments_dir: str = "experiments") -> list:
    """
    List all available experiment configs.
    
    Args:
        experiments_dir: Directory containing experiment configs
    
    Returns:
        List of experiment metadata dictionaries
    """
    if not os.path.exists(experiments_dir):
        print(f"Experiments directory not found: {experiments_dir}")
        return []
    
    configs = []
    for root, dirs, files in os.walk(experiments_dir):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                config_path = os.path.join(root, file)
                try:
                    omega_config = OmegaConf.load(config_path)
                    exp_info = {
                        'file': os.path.relpath(config_path, experiments_dir),
                        'full_path': config_path,
                        'name': file.replace('.yaml', '').replace('.yml', ''),
                        'description': 'N/A'
                    }
                    
                    # Try to extract experiment metadata
                    if 'experiment' in omega_config:
                        exp_info['name'] = omega_config.experiment.get('name', exp_info['name'])
                        exp_info['description'] = omega_config.experiment.get('description', 'N/A')
                    
                    configs.append(exp_info)
                except Exception as e:
                    print(f"Warning: Failed to load {config_path}: {e}")
    
    return configs


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
    
    Returns:
        Merged configuration dictionary
    """
    merged = OmegaConf.create({})
    for config in configs:
        if isinstance(config, (dict, DictConfig)):
            merged = OmegaConf.merge(merged, config)
    return merged


if __name__ == "__main__":
    # Example usage
    config = load_experiment_config("experiments/video_frequency_test/high_res_25fps.yaml")
    assert config != wm_orca_args()

    # list all experiments
    experiments = list_available_experiments()
    for exp in experiments:
        print(f"Experiment: {exp['name']}, Description: {exp['description']}, File: {exp['file']}")

    # save config to yaml
    save_config_to_yaml(config, "output/saved_config.yaml")