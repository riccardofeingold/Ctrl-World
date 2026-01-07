"""
Script to analyze video predictions and generate pixel error heatmap evolution.

The input video should have:
- Upper half: Ground truth frames
- Lower half: Predicted frames

The script will:
1. Split each frame into ground truth and prediction
2. Calculate pixel-wise error (MSE or MAE)
3. Generate heatmap evolution showing error over time
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Tuple, List
import seaborn as sns


def load_video(video_path: str) -> Tuple[np.ndarray, int, int]:
    """
    Load video and return frames as numpy array.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        frames: Array of shape (num_frames, height, width, channels)
        fps: Frames per second
        total_frames: Total number of frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    return np.array(frames), fps, total_frames


def split_frames(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split frames into ground truth (upper half) and predictions (lower half).
    
    Args:
        frames: Array of shape (num_frames, height, width, channels)
        
    Returns:
        gt_frames: Ground truth frames (upper half)
        pred_frames: Predicted frames (lower half)
    """
    num_frames, height, width, channels = frames.shape
    mid_height = height // 2
    
    gt_frames = frames[:, :mid_height, :, :]
    pred_frames = frames[:, mid_height:, :, :]
    
    return gt_frames, pred_frames


def calculate_pixel_error(gt_frames: np.ndarray, pred_frames: np.ndarray, 
                         error_type: str = 'mse') -> np.ndarray:
    """
    Calculate pixel-wise error between ground truth and predictions.
    
    Args:
        gt_frames: Ground truth frames
        pred_frames: Predicted frames
        error_type: Type of error metric ('mse' or 'mae')
        
    Returns:
        errors: Array of shape (num_frames, height, width) with error values
    """
    # Convert to float for calculation
    gt_frames = gt_frames.astype(np.float32)
    pred_frames = pred_frames.astype(np.float32)
    
    if error_type == 'mse':
        # Mean squared error across color channels
        errors = np.mean((gt_frames - pred_frames) ** 2, axis=-1)
    elif error_type == 'mae':
        # Mean absolute error across color channels
        errors = np.mean(np.abs(gt_frames - pred_frames), axis=-1)
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    return errors


def plot_heatmap_evolution(errors: np.ndarray, output_path: str, fps: int = 30,
                          cmap: str = 'hot', interval: int = 100,
                          vmin: float = None, vmax: float = None):
    """
    Create an animated heatmap showing error evolution over time.
    
    Args:
        errors: Array of shape (num_frames, height, width) with error values
        output_path: Path to save the animation (e.g., 'output.mp4' or 'output.gif')
        fps: Frames per second for the animation
        cmap: Colormap to use for heatmap
        interval: Delay between frames in milliseconds
        vmin: Minimum value for colormap (None for auto)
        vmax: Maximum value for colormap (None for auto)
    """
    num_frames, height, width = errors.shape
    
    # Set colormap limits if not provided
    if vmin is None:
        vmin = np.percentile(errors, 1)  # Use 1st percentile to avoid outliers
    if vmax is None:
        vmax = np.percentile(errors, 99)  # Use 99th percentile to avoid outliers
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize with first frame
    im = ax.imshow(errors[0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel Error', rotation=270, labelpad=20)
    
    title = ax.set_title(f'Pixel Error Heatmap - Frame 0/{num_frames}', fontsize=14)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    
    # Add statistics text
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.5), fontsize=10)
    
    def update(frame_idx):
        """Update function for animation."""
        im.set_data(errors[frame_idx])
        title.set_text(f'Pixel Error Heatmap - Frame {frame_idx}/{num_frames}')
        
        # Update statistics
        mean_error = np.mean(errors[frame_idx])
        max_error = np.max(errors[frame_idx])
        min_error = np.min(errors[frame_idx])
        std_error = np.std(errors[frame_idx])
        
        stats_text.set_text(
            f'Mean: {mean_error:.2f}\n'
            f'Std: {std_error:.2f}\n'
            f'Min: {min_error:.2f}\n'
            f'Max: {max_error:.2f}'
        )
        
        return im, title, stats_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=interval, blit=True, repeat=True)
    
    # Save animation
    output_path = Path(output_path)
    if output_path.suffix == '.gif':
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(output_path), writer=writer)
    
    plt.close()
    print(f"Animation saved to: {output_path}")


def plot_error_statistics(errors: np.ndarray, output_path: str):
    """
    Plot error statistics over time.
    
    Args:
        errors: Array of shape (num_frames, height, width) with error values
        output_path: Path to save the plot
    """
    num_frames = errors.shape[0]
    
    # Calculate statistics per frame
    mean_errors = np.mean(errors, axis=(1, 2))
    std_errors = np.std(errors, axis=(1, 2))
    max_errors = np.max(errors, axis=(1, 2))
    min_errors = np.min(errors, axis=(1, 2))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    frames = np.arange(num_frames)
    
    # Mean error over time
    axes[0, 0].plot(frames, mean_errors, linewidth=2, color='blue')
    axes[0, 0].fill_between(frames, mean_errors - std_errors, mean_errors + std_errors, 
                            alpha=0.3, color='blue')
    axes[0, 0].set_title('Mean Pixel Error Over Time', fontsize=12)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Mean Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard deviation over time
    axes[0, 1].plot(frames, std_errors, linewidth=2, color='orange')
    axes[0, 1].set_title('Error Standard Deviation Over Time', fontsize=12)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Std Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Max error over time
    axes[1, 0].plot(frames, max_errors, linewidth=2, color='red')
    axes[1, 0].set_title('Maximum Pixel Error Over Time', fontsize=12)
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Max Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution histogram
    axes[1, 1].hist(errors.flatten(), bins=100, color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Overall Error Distribution', fontsize=12)
    axes[1, 1].set_xlabel('Error Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Statistics plot saved to: {output_path}")


def plot_spatial_error_map(errors: np.ndarray, output_path: str):
    """
    Plot average spatial error map across all frames.
    
    Args:
        errors: Array of shape (num_frames, height, width) with error values
        output_path: Path to save the plot
    """
    # Average error across all frames
    avg_error = np.mean(errors, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(avg_error, cmap='hot', interpolation='bilinear')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Pixel Error', rotation=270, labelpad=20)
    
    ax.set_title('Average Spatial Error Distribution', fontsize=14)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spatial error map saved to: {output_path}")


def find_checkpoint_videos(base_dir: str, checkpoint: str = None) -> List[Tuple[str, str]]:
    """
    Find all videos matching the checkpoint pattern in the base directory structure.
    
    Args:
        base_dir: Base directory containing model folders
        checkpoint: Checkpoint pattern to match (e.g., 'train_steps_82505', 'train_steps_*')
                   If None, returns all videos
        
    Returns:
        List of tuples (model_folder_name, video_path)
    """
    import os
    import glob
    
    videos = []
    base_path = Path(base_dir)
    
    # Get all subdirectories in base_dir
    for model_folder in sorted(base_path.iterdir()):
        if not model_folder.is_dir():
            continue
            
        samples_dir = model_folder / "samples"
        if not samples_dir.exists():
            continue
        
        # Find videos matching the checkpoint pattern
        if checkpoint is None:
            # Get all videos
            pattern = str(samples_dir / "*.mp4")
        else:
            # Match specific checkpoint
            if not checkpoint.endswith("*.mp4"):
                if not checkpoint.endswith(".mp4"):
                    pattern = str(samples_dir / f"{checkpoint}_*.mp4")
                else:
                    pattern = str(samples_dir / checkpoint)
            else:
                pattern = str(samples_dir / checkpoint)
        
        matching_videos = glob.glob(pattern)
        for video_path in matching_videos:
            videos.append((model_folder.name, video_path))
    
    return videos


def select_folders(available_folders: List[str]) -> List[str]:
    """
    Let user select which folders to analyze.
    
    Args:
        available_folders: List of available folder names
        
    Returns:
        List of selected folder names
    """
    print("\nAvailable model folders:")
    for idx, folder in enumerate(available_folders, 1):
        print(f"  {idx}. {folder}")
    
    print("\nSelect folders to analyze:")
    print("  - Enter folder numbers separated by commas (e.g., 1,3,5)")
    print("  - Enter 'all' to analyze all folders")
    print("  - Enter ranges with dash (e.g., 1-3,5)")
    
    selection = input("Your selection: ").strip()
    
    if selection.lower() == 'all':
        return available_folders
    
    selected_indices = set()
    for part in selection.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            selected_indices.update(range(int(start), int(end) + 1))
        else:
            selected_indices.add(int(part))
    
    selected_folders = [available_folders[i-1] for i in sorted(selected_indices) 
                       if 0 < i <= len(available_folders)]
    
    return selected_folders


def process_video(video_path: str, output_dir: Path, args):
    """Process a single video and generate all outputs."""
    # Load video
    print(f"\n{'='*80}")
    print(f"Processing: {video_path}")
    print(f"{'='*80}")
    frames, video_fps, total_frames = load_video(video_path)
    print(f"Loaded {total_frames} frames at {video_fps} FPS")
    print(f"Frame shape: {frames.shape}")
    
    # Use video FPS if not specified
    fps = args.fps if args.fps is not None else video_fps
    
    # Split frames into ground truth and predictions
    print("Splitting frames into ground truth and predictions...")
    gt_frames, pred_frames = split_frames(frames)
    print(f"Ground truth shape: {gt_frames.shape}")
    print(f"Predictions shape: {pred_frames.shape}")
    
    # Calculate pixel errors
    print(f"Calculating pixel errors using {args.error_type.upper()}...")
    errors = calculate_pixel_error(gt_frames, pred_frames, error_type=args.error_type)
    print(f"Error array shape: {errors.shape}")
    print(f"Mean error: {np.mean(errors):.2f}")
    print(f"Max error: {np.max(errors):.2f}")
    
    # Generate base filename
    video_name = Path(video_path).stem
    
    # Plot heatmap evolution
    print("Generating heatmap evolution animation...")
    anim_path = output_dir / f"{video_name}_heatmap_evolution.{args.format}"
    plot_heatmap_evolution(errors, str(anim_path), fps=fps, 
                          cmap=args.cmap, interval=args.interval,
                          vmin=args.vmin, vmax=args.vmax)
    
    # Plot error statistics
    print("Generating error statistics plot...")
    stats_path = output_dir / f"{video_name}_error_statistics.png"
    plot_error_statistics(errors, str(stats_path))
    
    # Plot spatial error map
    print("Generating spatial error map...")
    spatial_path = output_dir / f"{video_name}_spatial_error.png"
    plot_spatial_error_map(errors, str(spatial_path))
    
    print(f"Outputs saved to: {output_dir}")


def main():
    import os

    parser = argparse.ArgumentParser(
        description='Analyze video predictions and generate pixel error heatmap evolution.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single video
  python script.py path/to/video.mp4

  # Analyze videos from multiple model folders with specific checkpoint
  python script.py --base_dir model_ckpt --checkpoint train_steps_82505

  # Analyze with specific checkpoint pattern (non-interactive)
  python script.py --base_dir model_ckpt --checkpoint "train_steps_82505_0" --folders "folder1,folder2"
        """
    )
    
    # Main input: either video_path or base_dir
    parser.add_argument('video_path', type=str, nargs='?', default=None,
                       help='Path to a single input video file (optional if --base_dir is used)')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='Base directory containing model folders with samples subdirectories')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint pattern to match (e.g., "train_steps_82505" or "train_steps_82505_0")')
    parser.add_argument('--folders', type=str, default=None,
                       help='Comma-separated list of folder names to analyze (default: interactive selection)')
    
    # Output and processing options
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save output files (default: auto-generated)')
    parser.add_argument('--error_type', type=str, default='mse', choices=['mse', 'mae'],
                       help='Type of error metric (default: mse)')
    parser.add_argument('--cmap', type=str, default='hot',
                       help='Colormap for heatmap (default: hot)')
    parser.add_argument('--fps', type=int, default=None,
                       help='FPS for output animation (default: same as input)')
    parser.add_argument('--format', type=str, default='mp4', choices=['mp4', 'gif'],
                       help='Output format for animation (default: mp4)')
    parser.add_argument('--interval', type=int, default=100,
                       help='Delay between frames in milliseconds (default: 100)')
    parser.add_argument('--vmin', type=float, default=None,
                       help='Minimum value for colormap (default: auto)')
    parser.add_argument('--vmax', type=float, default=None,
                       help='Maximum value for colormap (default: auto)')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if args.video_path is None and args.base_dir is None:
        parser.error("Either video_path or --base_dir must be provided")
    
    if args.video_path is not None and args.base_dir is not None:
        parser.error("Cannot specify both video_path and --base_dir")
    
    # Mode 1: Single video processing
    if args.video_path is not None:
        # Create output directory
        if args.output_dir is None:
            output_dir = Path(args.video_path).parent / 'heatmap_analysis_output'
        else:
            output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        process_video(args.video_path, output_dir, args)
        print("\nAnalysis complete!")
        return
    
    # Mode 2: Batch processing from base_dir
    print(f"Searching for videos in: {args.base_dir}")
    
    # Find all videos
    all_videos = find_checkpoint_videos(args.base_dir, args.checkpoint)
    
    if not all_videos:
        print(f"No videos found in {args.base_dir}")
        if args.checkpoint:
            print(f"with checkpoint pattern: {args.checkpoint}")
        return
    
    # Group videos by model folder
    videos_by_folder = {}
    for folder_name, video_path in all_videos:
        if folder_name not in videos_by_folder:
            videos_by_folder[folder_name] = []
        videos_by_folder[folder_name].append(video_path)
    
    available_folders = sorted(videos_by_folder.keys())
    print(f"\nFound {len(available_folders)} model folders with videos")
    
    # Select folders to process
    if args.folders is not None:
        # Non-interactive: use provided folder names
        selected_folders = [f.strip() for f in args.folders.split(',')]
        # Validate folder names
        invalid_folders = [f for f in selected_folders if f not in available_folders]
        if invalid_folders:
            print(f"Warning: Unknown folders will be skipped: {invalid_folders}")
        selected_folders = [f for f in selected_folders if f in available_folders]
    else:
        # Interactive selection
        selected_folders = select_folders(available_folders)
    
    if not selected_folders:
        print("No folders selected. Exiting.")
        return
    
    print(f"\nWill analyze {len(selected_folders)} folder(s):")
    for folder in selected_folders:
        num_videos = len(videos_by_folder[folder])
        print(f"  - {folder}: {num_videos} video(s)")
    
    # Process all videos in selected folders
    total_videos = sum(len(videos_by_folder[f]) for f in selected_folders)
    processed = 0
    
    for folder_name in selected_folders:
        videos = videos_by_folder[folder_name]
        
        for video_path in videos:
            processed += 1
            print(f"\n[{processed}/{total_videos}] Processing {folder_name}")
            
            # Create output directory for this model folder
            if args.output_dir is None:
                output_dir = Path(args.base_dir) / folder_name / 'heatmap_analysis'
            else:
                output_dir = Path(args.output_dir) / folder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                process_video(video_path, output_dir, args)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
    
    print(f"\n{'='*80}")
    print(f"Batch analysis complete!")
    print(f"Processed {processed} video(s) from {len(selected_folders)} folder(s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
