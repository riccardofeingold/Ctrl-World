#!/usr/bin/env python3
"""
Comprehensive statistical analysis of JSON trajectory data.
Analyzes observations, actions, and their relationships across time.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import pandas as pd
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def extract_field_data(data: Dict, field_name: str) -> np.ndarray:
    """Extract and convert field data to numpy array."""
    if field_name in data:
        return np.array(data[field_name])
    else:
        print(f"Warning: Field '{field_name}' not found in data")
        return np.array([])


def compute_statistics(arr: np.ndarray, field_name: str) -> Dict[str, Any]:
    """Compute comprehensive statistics for a given array."""
    if arr.size == 0:
        return {}
    
    # Reshape to (time_steps, dimensions)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    
    stats_dict = {
        'field_name': field_name,
        'shape': arr.shape,
        'n_timesteps': arr.shape[0],
        'n_dimensions': arr.shape[1] if arr.ndim > 1 else 1,
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0),
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'median': np.median(arr, axis=0),
        'percentile_25': np.percentile(arr, 25, axis=0),
        'percentile_75': np.percentile(arr, 75, axis=0),
        'percentile_1': np.percentile(arr, 1, axis=0),
        'percentile_99': np.percentile(arr, 99, axis=0),
        'range': np.max(arr, axis=0) - np.min(arr, axis=0),
        'variance': np.var(arr, axis=0),
        'skewness': stats.skew(arr, axis=0),
        'kurtosis': stats.kurtosis(arr, axis=0),
    }
    
    # Add temporal statistics (change over time)
    if arr.shape[0] > 1:
        diff = np.diff(arr, axis=0)
        stats_dict['temporal_change_mean'] = np.mean(diff, axis=0)
        stats_dict['temporal_change_std'] = np.std(diff, axis=0)
        stats_dict['temporal_change_max'] = np.max(np.abs(diff), axis=0)
    
    return stats_dict


def print_statistics(stats_dict: Dict[str, Any]):
    """Print statistics in a readable format."""
    if not stats_dict:
        return
    
    print(f"\n{'='*80}")
    print(f"Statistics for: {stats_dict['field_name']}")
    print(f"{'='*80}")
    print(f"Shape: {stats_dict['shape']}")
    print(f"Time steps: {stats_dict['n_timesteps']}")
    print(f"Dimensions: {stats_dict['n_dimensions']}")
    print(f"\nPer-dimension statistics:")
    print(f"{'-'*80}")
    
    # Create a table for better readability
    metrics = ['mean', 'std', 'min', 'max', 'median', 'percentile_1', 'percentile_99', 
               'range', 'variance', 'skewness', 'kurtosis']
    
    for metric in metrics:
        if metric in stats_dict:
            values = stats_dict[metric]
            if isinstance(values, np.ndarray):
                if len(values) <= 6:  # Print all if small
                    print(f"{metric:20s}: {values}")
                else:  # Print summary if large
                    print(f"{metric:20s}: min={values.min():.4f}, max={values.max():.4f}, "
                          f"mean={values.mean():.4f}, std={values.std():.4f}")
    
    # Temporal statistics
    if 'temporal_change_mean' in stats_dict:
        print(f"\nTemporal change statistics (derivatives):")
        print(f"{'-'*80}")
        print(f"{'Mean change':20s}: {stats_dict['temporal_change_mean']}")
        print(f"{'Std of change':20s}: {stats_dict['temporal_change_std']}")
        print(f"{'Max abs change':20s}: {stats_dict['temporal_change_max']}")


def plot_time_series(arr: np.ndarray, field_name: str, output_dir: Path):
    """Plot time series for each dimension."""
    if arr.size == 0:
        return
    
    n_dims = arr.shape[1] if arr.ndim > 1 else 1
    n_timesteps = arr.shape[0]
    
    # Determine subplot layout
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    if n_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_dims):
        ax = axes[i]
        data = arr[:, i] if arr.ndim > 1 else arr
        ax.plot(data, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{field_name} - Dimension {i}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    safe_name = field_name.replace('.', '_').replace(' ', '_')
    plt.savefig(output_dir / f'{safe_name}_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_distributions(arr: np.ndarray, field_name: str, output_dir: Path):
    """Plot distributions for each dimension."""
    if arr.size == 0:
        return
    
    n_dims = arr.shape[1] if arr.ndim > 1 else 1
    
    # Determine subplot layout
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    if n_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_dims):
        ax = axes[i]
        data = arr[:, i] if arr.ndim > 1 else arr
        
        # Histogram with KDE
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', density=True)
        
        # Add KDE if we have enough data points and variance
        if len(data) > 3 and np.std(data) > 1e-10:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except (np.linalg.LinAlgError, ValueError):
                # Skip KDE for constant or near-constant data
                pass
        
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{field_name} - Dim {i} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    safe_name = field_name.replace('.', '_').replace(' ', '_')
    plt.savefig(output_dir / f'{safe_name}_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(arr: np.ndarray, arr2: np.ndarray, field_name: str, output_dir: Path):
    """Plot correlation matrix between dimensions."""
    if arr.size == 0 or arr.shape[1] <= 1:
        return
    
    corr_matrix = np.corrcoef(arr.T, arr2.T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Matrix: {field_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension', fontsize=12)
    plt.ylabel('Dimension', fontsize=12)
    plt.tight_layout()
    
    safe_name = field_name.replace('.', '_').replace(' ', '_')
    plt.savefig(output_dir / f'{safe_name}_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplot(arr: np.ndarray, field_name: str, output_dir: Path):
    """Plot boxplot for each dimension to show outliers."""
    if arr.size == 0:
        return
    
    n_dims = arr.shape[1] if arr.ndim > 1 else 1
    
    plt.figure(figsize=(max(15, n_dims), 8))
    
    # Create DataFrame for seaborn
    df_data = []
    for i in range(n_dims):
        data = arr[:, i] if arr.ndim > 1 else arr
        for val in data:
            df_data.append({'Dimension': f'Dim {i}', 'Value': val})
    
    df = pd.DataFrame(df_data)
    
    sns.boxplot(data=df, x='Dimension', y='Value')
    plt.title(f'Box Plot: {field_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    safe_name = field_name.replace('.', '_').replace(' ', '_')
    plt.savefig(output_dir / f'{safe_name}_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_action_observation_comparison(action_arr: np.ndarray, obs_arr: np.ndarray, 
                                      action_name: str, obs_name: str, output_dir: Path):
    """Compare action and corresponding observation at each time step."""
    if action_arr.size == 0 or obs_arr.size == 0:
        return
    
    # Handle different lengths
    min_len = min(action_arr.shape[0], obs_arr.shape[0])
    action_arr = action_arr[:min_len]
    obs_arr = obs_arr[:min_len]
    
    # Handle different dimensions
    min_dims = min(action_arr.shape[1], obs_arr.shape[1])
    
    # Determine subplot layout
    n_cols = min(3, min_dims)
    n_rows = (min_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if min_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(min_dims):
        ax = axes[i]
        ax.plot(action_arr[:, i], label=f'{action_name}', linewidth=2, marker='o', markersize=4)
        ax.plot(obs_arr[:, i], label=f'{obs_name}', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'Dimension {i}: Action vs Observation', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(min_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    safe_name = f"{action_name.replace('.', '_')}_{obs_name.replace('.', '_')}_comparison"
    plt.savefig(output_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_action_observation_scatter(action_arr: np.ndarray, obs_arr: np.ndarray,
                                    action_name: str, obs_name: str, output_dir: Path):
    """Scatter plot of action vs observation values to see correlation."""
    if action_arr.size == 0 or obs_arr.size == 0:
        return
    
    # Handle different lengths
    min_len = min(action_arr.shape[0], obs_arr.shape[0])
    action_arr = action_arr[:min_len]
    obs_arr = obs_arr[:min_len]
    
    # Handle different dimensions
    min_dims = min(action_arr.shape[1], obs_arr.shape[1])
    
    # Determine subplot layout
    n_cols = min(3, min_dims)
    n_rows = (min_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if min_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(min_dims):
        ax = axes[i]
        ax.scatter(obs_arr[:, i], action_arr[:, i], alpha=0.6, s=50)
        
        # Add regression line
        z = np.polyfit(obs_arr[:, i], action_arr[:, i], 1)
        p = np.poly1d(z)
        x_line = np.linspace(obs_arr[:, i].min(), obs_arr[:, i].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Calculate correlation
        corr = np.corrcoef(obs_arr[:, i], action_arr[:, i])[0, 1]
        
        ax.set_xlabel(f'{obs_name}', fontsize=10)
        ax.set_ylabel(f'{action_name}', fontsize=10)
        ax.set_title(f'Dim {i}: Correlation = {corr:.3f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(min_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    safe_name = f"{action_name.replace('.', '_')}_{obs_name.replace('.', '_')}_scatter"
    plt.savefig(output_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_temporal_changes(arr: np.ndarray, field_name: str, output_dir: Path):
    """Plot temporal changes (derivatives) over time."""
    if arr.size == 0 or arr.shape[0] <= 1:
        return
    
    diff = np.diff(arr, axis=0)
    n_dims = diff.shape[1] if diff.ndim > 1 else 1
    
    # Determine subplot layout
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    if n_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_dims):
        ax = axes[i]
        data = diff[:, i] if diff.ndim > 1 else diff
        ax.plot(data, linewidth=2, marker='o', markersize=4, color='green')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Change (Δ)', fontsize=10)
        ax.set_title(f'{field_name} - Dim {i} Temporal Change', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    safe_name = field_name.replace('.', '_').replace(' ', '_')
    plt.savefig(output_dir / f'{safe_name}_temporal_changes.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_field(data: Dict, field_name: str, output_dir: Path) -> Dict[str, Any]:
    """Perform comprehensive analysis on a single field."""
    print(f"\nAnalyzing field: {field_name}")
    
    arr = extract_field_data(data, field_name)
    
    if arr.size == 0:
        print(f"  -> Skipped (no data found)")
        return {}
    
    # Compute statistics
    stats = compute_statistics(arr, field_name)
    print_statistics(stats)
    
    # Generate plots
    plot_time_series(arr, field_name, output_dir)
    plot_distributions(arr, field_name, output_dir)
    plot_boxplot(arr, field_name, output_dir)
    plot_temporal_changes(arr, field_name, output_dir)
    
    return stats


def save_statistics_report(all_stats: Dict[str, Dict], output_dir: Path):
    """Save comprehensive statistics report to text file."""
    report_path = output_dir / 'statistics_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE STATISTICAL ANALYSIS REPORT\n")
        f.write("="*100 + "\n\n")
        
        for field_name, stats in all_stats.items():
            if not stats:
                continue
            
            f.write("\n" + "="*100 + "\n")
            f.write(f"Field: {field_name}\n")
            f.write("="*100 + "\n")
            f.write(f"Shape: {stats['shape']}\n")
            f.write(f"Time steps: {stats['n_timesteps']}\n")
            f.write(f"Dimensions: {stats['n_dimensions']}\n\n")
            
            f.write("Per-dimension Statistics:\n")
            f.write("-"*100 + "\n")
            
            for key in ['mean', 'std', 'min', 'max', 'median', 'percentile_1', 
                       'percentile_99', 'range', 'variance', 'skewness', 'kurtosis']:
                if key in stats:
                    f.write(f"{key:20s}: {stats[key]}\n")
            
            if 'temporal_change_mean' in stats:
                f.write("\nTemporal Change Statistics:\n")
                f.write("-"*100 + "\n")
                f.write(f"Mean change: {stats['temporal_change_mean']}\n")
                f.write(f"Std of change: {stats['temporal_change_std']}\n")
                f.write(f"Max abs change: {stats['temporal_change_max']}\n")
            
            f.write("\n")
    
    print(f"\nStatistics report saved to: {report_path}")


def compute_aggregate_statistics(all_data: List[Dict], fields: List[str]) -> Dict[str, Dict[str, Any]]:
    """Compute statistics aggregated across all JSON files."""
    aggregate_stats = {}
    
    for field in fields:
        print(f"\nComputing aggregate statistics for: {field}")
        
        # Collect all data for this field across all files
        all_arrays = []
        for data in all_data:
            arr = extract_field_data(data, field)
            if arr.size > 0:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                all_arrays.append(arr)
        
        if not all_arrays:
            print(f"  -> No data found across all files")
            continue
        
        # Concatenate all arrays along time dimension
        combined = np.concatenate(all_arrays, axis=0)
        
        # Compute statistics
        stats = {
            'field_name': field,
            'total_samples': combined.shape[0],
            'n_files': len(all_arrays),
            'n_dimensions': combined.shape[1] if combined.ndim > 1 else 1,
            'global_mean': np.mean(combined, axis=0),
            'global_std': np.std(combined, axis=0),
            'global_min': np.min(combined, axis=0),
            'global_max': np.max(combined, axis=0),
            'global_median': np.median(combined, axis=0),
            'global_percentile_1': np.percentile(combined, 1, axis=0),
            'global_percentile_99': np.percentile(combined, 99, axis=0),
            'global_range': np.max(combined, axis=0) - np.min(combined, axis=0),
            'global_variance': np.var(combined, axis=0),
        }
        
        # Compute mean and std of means across files (inter-trajectory statistics)
        file_means = np.array([np.mean(arr, axis=0) for arr in all_arrays])
        file_stds = np.array([np.std(arr, axis=0) for arr in all_arrays])
        
        stats['mean_of_means'] = np.mean(file_means, axis=0)
        stats['std_of_means'] = np.std(file_means, axis=0)
        stats['mean_of_stds'] = np.mean(file_stds, axis=0)
        stats['std_of_stds'] = np.std(file_stds, axis=0)
        
        aggregate_stats[field] = stats
        
        # Print summary
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Files analyzed: {stats['n_files']}")
        print(f"  Dimensions: {stats['n_dimensions']}")
        print(f"  Global mean: {stats['global_mean']}")
        print(f"  Global std: {stats['global_std']}")
    
    return aggregate_stats


def save_aggregate_statistics_report(aggregate_stats: Dict[str, Dict], output_dir: Path):
    """Save aggregate statistics report across all files."""
    report_path = output_dir / 'aggregate_statistics_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("AGGREGATE STATISTICS ACROSS ALL JSON FILES\n")
        f.write("="*100 + "\n\n")
        
        for field_name, stats in aggregate_stats.items():
            f.write("\n" + "="*100 + "\n")
            f.write(f"Field: {field_name}\n")
            f.write("="*100 + "\n")
            f.write(f"Total samples across all files: {stats['total_samples']}\n")
            f.write(f"Number of files: {stats['n_files']}\n")
            f.write(f"Dimensions: {stats['n_dimensions']}\n\n")
            
            f.write("Global Statistics (across all samples):\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Global mean':25s}: {stats['global_mean']}\n")
            f.write(f"{'Global std':25s}: {stats['global_std']}\n")
            f.write(f"{'Global min':25s}: {stats['global_min']}\n")
            f.write(f"{'Global max':25s}: {stats['global_max']}\n")
            f.write(f"{'Global median':25s}: {stats['global_median']}\n")
            f.write(f"{'Global percentile 1%':25s}: {stats['global_percentile_1']}\n")
            f.write(f"{'Global percentile 99%':25s}: {stats['global_percentile_99']}\n")
            f.write(f"{'Global range':25s}: {stats['global_range']}\n")
            f.write(f"{'Global variance':25s}: {stats['global_variance']}\n\n")
            
            f.write("Inter-trajectory Statistics (variation across files):\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Mean of file means':25s}: {stats['mean_of_means']}\n")
            f.write(f"{'Std of file means':25s}: {stats['std_of_means']}\n")
            f.write(f"{'Mean of file stds':25s}: {stats['mean_of_stds']}\n")
            f.write(f"{'Std of file stds':25s}: {stats['std_of_stds']}\n\n")
    
    print(f"\nAggregate statistics report saved to: {report_path}")


def plot_aggregate_distributions(all_data: List[Dict], fields: List[str], output_dir: Path):
    """Plot aggregate distributions across all files."""
    for field in fields:
        print(f"\nPlotting aggregate distribution for: {field}")
        
        # Collect all data
        all_arrays = []
        for data in all_data:
            arr = extract_field_data(data, field)
            if arr.size > 0:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                all_arrays.append(arr)
        
        if not all_arrays:
            continue
        
        combined = np.concatenate(all_arrays, axis=0)
        n_dims = combined.shape[1]
        
        # Determine subplot layout
        n_cols = min(4, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        if n_dims == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i in range(n_dims):
            ax = axes[i]
            data = combined[:, i]
            
            # Histogram with KDE
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black', density=True)
            
            # Add KDE if we have enough data points and variance
            if len(data) > 3 and np.std(data) > 1e-10:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                except (np.linalg.LinAlgError, ValueError):
                    pass
            
            # Add statistics
            mean = np.mean(data)
            std = np.std(data)
            ax.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Mean={mean:.3f}')
            ax.axvline(mean + std, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax.axvline(mean - std, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'±1 Std')
            
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{field} - Dim {i} (N={len(data)})', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_dims, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        safe_name = field.replace('.', '_').replace(' ', '_')
        plt.savefig(output_dir / f'{safe_name}_aggregate_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main analysis function."""
    import argparse
    import random
    import glob
    
    parser = argparse.ArgumentParser(description='Analyze JSON trajectory data')
    parser.add_argument('--json_file', type=str, 
                       default=None,
                       help='Path to a single JSON file (deprecated, use --json_dir instead)')
    parser.add_argument('--json_dir', type=str,
                       default='datasets/orca_D1/annotation/train',
                       help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory to save analysis results')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of random JSON files to sample for detailed analysis')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--compute_aggregate', action='store_true', default=True,
                       help='Compute aggregate statistics across all files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Fields to analyze
    fields = [
        'states',
        'observation.state.joint_position',
        'observation.state.cartesian_position',
        'observation.state.hand_joint_position',
        'action.cartesian_position',
        'action.hand_joint_position',
    ]
    
    # Determine which mode to use
    if args.json_file:
        # Single file mode (backward compatibility)
        print(f"Loading single file: {args.json_file}")
        json_files = [args.json_file]
    else:
        # Directory mode
        json_dir = Path(args.json_dir)
        if not json_dir.exists():
            print(f"Error: Directory '{args.json_dir}' does not exist")
            return
        
        # Find all JSON files
        json_files = sorted(glob.glob(str(json_dir / "*.json")))
        
        if not json_files:
            print(f"Error: No JSON files found in '{args.json_dir}'")
            return
        
        print(f"Found {len(json_files)} JSON files in {args.json_dir}")
        
        # Randomly sample files for detailed analysis
        random.seed(args.random_seed)
        if len(json_files) > args.n_samples:
            sampled_files = random.sample(json_files, args.n_samples)
            print(f"Randomly selected {args.n_samples} files for detailed analysis")
        else:
            sampled_files = json_files
            print(f"Using all {len(json_files)} files for detailed analysis")
    
    # Load all data for aggregate statistics
    print("\n" + "="*80)
    print("Loading all JSON files for aggregate statistics...")
    print("="*80)
    all_data = []
    for i, json_file in enumerate(json_files):
        try:
            data = load_json_data(json_file)
            all_data.append(data)
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(json_files)} files...")
        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")
    
    print(f"Successfully loaded {len(all_data)} JSON files")
    
    # Perform detailed analysis on sampled files
    if args.json_file:
        sampled_files = json_files
    
    for idx, json_file in enumerate(sampled_files):
        print("\n" + "="*80)
        print(f"Detailed Analysis of File {idx + 1}/{len(sampled_files)}: {Path(json_file).name}")
        print("="*80)
        
        # Create subdirectory for this file
        file_output_dir = output_dir / f"sample_{idx}_{Path(json_file).stem}"
        file_output_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            data = load_json_data(json_file)
            
            # Analyze each field
            all_stats = {}
            for field in fields:
                stats = analyze_field(data, field, file_output_dir)
                if stats:
                    all_stats[field] = stats
            
            # Save statistics report for this file
            save_statistics_report(all_stats, file_output_dir)
            
            # Compare actions with observations
            print("\nComparing Actions with Observations")
            print("-"*80)
            
            # Cartesian position comparison
            action_cart = extract_field_data(data, 'action.cartesian_position')
            obs_cart = extract_field_data(data, 'observation.state.cartesian_position')
            if action_cart.size > 0 and obs_cart.size > 0:
                print("  Comparing cartesian positions...")
                plot_action_observation_comparison(
                    action_cart, obs_cart,
                    'Action Cartesian', 'Obs Cartesian',
                    file_output_dir
                )
                plot_action_observation_scatter(
                    action_cart, obs_cart,
                    'Action Cartesian', 'Obs Cartesian',
                    file_output_dir
                )
                plot_correlation_matrix(
                    action_cart, obs_cart,
                    'Action vs Observation Cartesian Position',
                    file_output_dir
                )
            
            # Hand joint position comparison
            action_hand = extract_field_data(data, 'action.hand_joint_position')
            obs_hand = extract_field_data(data, 'observation.state.hand_joint_position')
            if action_hand.size > 0 and obs_hand.size > 0:
                print("  Comparing hand joint positions...")
                plot_action_observation_comparison(
                    action_hand, obs_hand,
                    'Action Hand Joint', 'Obs Hand Joint',
                    file_output_dir
                )
                plot_action_observation_scatter(
                    action_hand, obs_hand,
                    'Action Hand Joint', 'Obs Hand Joint',
                    file_output_dir
                )
                plot_correlation_matrix(
                    action_hand, obs_hand,
                    'Action vs Observation Hand Joint Position',
                    file_output_dir
                )
        
        except Exception as e:
            print(f"  Error analyzing {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute aggregate statistics across all files
    if args.compute_aggregate and len(all_data) > 0:
        print("\n" + "="*80)
        print("Computing Aggregate Statistics Across All Files")
        print("="*80)
        
        aggregate_stats = compute_aggregate_statistics(all_data, fields)
        save_aggregate_statistics_report(aggregate_stats, output_dir)
        
        # Plot aggregate distributions
        print("\n" + "="*80)
        print("Plotting Aggregate Distributions")
        print("="*80)
        plot_aggregate_distributions(all_data, fields, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  Main directory:")
    print(f"    - aggregate_statistics_report.txt: Statistics across all files")
    print(f"    - *_aggregate_distribution.png: Distribution plots across all files")
    print(f"\n  Per-sample directories (sample_0_*, sample_1_*, ...):")
    print(f"    - statistics_report.txt: Comprehensive text report for that file")
    print(f"    - *_timeseries.png: Time series plots for each field")
    print(f"    - *_distributions.png: Value distribution plots")
    print(f"    - *_correlation.png: Correlation matrices")
    print(f"    - *_boxplot.png: Box plots showing outliers")
    print(f"    - *_temporal_changes.png: Temporal derivative plots")
    print(f"    - *_comparison.png: Action vs observation comparisons")
    print(f"    - *_scatter.png: Scatter plots with correlation")


if __name__ == '__main__':
    main()
