"""Benchmarking script for evaluating world model performance with various metrics.

This module provides functionality to evaluate Ctrl-World model checkpoints on validation
datasets, computing metrics such as MSE, MAE, PSNR, SSIM, and ROI-specific metrics for
different semantic regions (hand, object, background).
"""
import time
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import einops
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from accelerate import Accelerator

from config import wm_orca_args
from utils.config_loader import load_experiment_config
from dataset.dataset_orca import Dataset_mix
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

from dotenv import load_dotenv
load_dotenv()

# For video processing and latent extraction
try:
    import mediapy
except Exception:
    mediapy = None
try:
    from diffusers.models import AutoencoderKLTemporalDecoder
except Exception:
    AutoencoderKLTemporalDecoder = None

# YAML parsing with interpolation support (same stack as training)
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

# Optional metric deps
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except Exception:
    ssim = None
    psnr = None

try:
    import lpips  # not required, but we may report if installed
except Exception:
    lpips = None

try:
    import torch_fidelity  # FID (image-level)
except Exception:
    torch_fidelity = None

# FVD usually needs a pretrained I3D model; try a common package
try:
    from pytorch_fid import inception
except Exception:
    inception = None

# For heatmap animations
try:
    import matplotlib.animation as animation
except Exception:
    animation = None

# =============================================================================
# Configuration for segmentation classes (pixel colors as RGB)
# Hand is defined as "everything not in these classes"
# =============================================================================
SEMANTIC_SEGMENTATION_MAPPING: Dict[str, Tuple[int, int, int]] = {
    "class:object": (0, 0, 255),
    "class:robot": (0, 255, 0),
    "class:ground": (255, 0, 0),
    "class:table": (255, 255, 0),
    "class:background": (0, 0, 0),
}


def find_closest_color_mask(seg_rgb: np.ndarray, target_color: Tuple[int, int, int], color_threshold: float = 15.0) -> np.ndarray:
    """
    Find pixels in segmentation RGB that match the target color within a threshold.
    
    Uses Euclidean distance in RGB space to handle video compression artifacts.
    
    Args:
        seg_rgb: Segmentation RGB array (T, H, W, 3) or (H, W, 3) uint8
        target_color: Target RGB color (R, G, B) as tuple
        color_threshold: Maximum Euclidean distance in RGB space to consider a match (default: 15.0)
    
    Returns:
        Boolean mask (T, H, W) or (H, W) indicating pixels matching the target color
    """
    seg = seg_rgb.astype(np.float32)  # Convert to float for distance calculation
    target = np.array(target_color, dtype=np.float32)
    
    # Calculate Euclidean distance in RGB space
    # seg: (T, H, W, 3) or (H, W, 3), target: (3,)
    # Broadcasting: subtract target from each pixel
    diff = seg - target  # (T, H, W, 3) or (H, W, 3)
    distances = np.linalg.norm(diff, axis=-1)  # (T, H, W) or (H, W)
    
    # Threshold: pixels with distance <= threshold are considered matches
    mask = distances <= color_threshold
    return mask.astype(bool)


def find_experiment_config_by_name(experiments_dir: str, experiment_name: str) -> Optional[str]:
    """
    Recursively search all *.yaml/*.yml under experiments_dir (including nested subfolders) and return:
    - an exact match on `experiment.experiment_name` if found
    - otherwise the best match (highest similarity) across:
      `experiment.experiment_name`, `training.tag`, and the YAML filename stem.

    This mirrors how experiment YAMLs are structured in this repo (see `experiments/base_config.yaml`).
    """
    if not os.path.exists(experiments_dir):
        return None
    if OmegaConf is None:
        raise ImportError("OmegaConf is required to scan experiment YAMLs. Please install omegaconf.")

    import difflib

    target = str(experiment_name).strip().lower()
    best_path: Optional[str] = None
    best_score: float = -1.0

    for root, _, files in os.walk(experiments_dir):
        for f in files:
            if not (f.endswith(".yaml") or f.endswith(".yml")):
                continue
            path = os.path.join(root, f)
            try:
                cfg = OmegaConf.load(path)
                if "experiment" not in cfg:
                    continue
                exp = cfg.experiment
                exp_name = str(exp.get("experiment_name", "")).strip()
                train_tag = ""
                try:
                    if "training" in cfg:
                        train_tag = str(cfg.training.get("tag", "")).strip()
                except Exception:
                    train_tag = ""

                # Exact match first
                if exp_name and exp_name.strip().lower() == target:
                    return path

                # Otherwise score and keep best match
                candidates = [
                    exp_name,
                    train_tag,
                    os.path.splitext(os.path.basename(f))[0],
                ]
                for cand in candidates:
                    cand_norm = str(cand).strip().lower()
                    if not cand_norm:
                        continue
                    score = difflib.SequenceMatcher(a=target, b=cand_norm).ratio()
                    # small bonuses for containment (helps tags that are prefixes/suffixes)
                    if target in cand_norm or cand_norm in target:
                        score += 0.15
                    if score > best_score:
                        best_score = score
                        best_path = path
            except Exception:
                # Skip unreadable configs
                continue

    # Require a minimal similarity so we don't return an unrelated config
    if best_path is not None and best_score >= 0.45:
        return best_path
    return None


def load_args_from_experiment_name(
    experiments_dir: str,
    experiment_name: str,
    base_args: Optional[wm_orca_args] = None,
) -> Tuple[wm_orca_args, str]:
    """
    Find experiment YAML by `experiment.experiment_name` and load it into wm_orca_args
    using the same loader as training (`utils.config_loader.load_experiment_config`).
    Returns (args, config_path).
    """
    config_path = find_experiment_config_by_name(experiments_dir, experiment_name)
    if config_path is None:
        raise FileNotFoundError(
            f"Could not find an experiment YAML with experiment.experiment_name='{experiment_name}' under '{experiments_dir}'."
        )
    args = load_experiment_config(config_path, base_args or wm_orca_args())
    return args, config_path


def resolve_ckpt_tag_and_args(
    name_or_tag: str,
    model_ckpt_root: str,
    dataset_root: str,
    experiments_dir: Optional[str] = None,
) -> Tuple[str, wm_orca_args, Optional[str]]:
    """
    Resolve a user-provided identifier into:
    - `tag`: the folder name under model_ckpt_root
    - `args`: fully-populated wm_orca_args (from YAML if found, else heuristic defaults)
    - `config_path`: the YAML path if resolved via experiments_dir, else None

    Resolution order:
    - If model_ckpt_root/name_or_tag exists as a directory, treat it as training.tag
    - Else if experiments_dir provided, treat name_or_tag as experiment.experiment_name and load YAML,
      then use args.tag (training.tag) to locate the checkpoint folder.
    """
    config_path = None
    ckpt_dir = os.path.join(model_ckpt_root, name_or_tag)
    if os.path.isdir(ckpt_dir):
        tag = name_or_tag
        args = make_args_for_tag(tag, dataset_root)
        return tag, args, None

    if experiments_dir is None:
        raise FileNotFoundError(
            f"Checkpoint folder '{ckpt_dir}' not found and no experiments_dir provided to resolve experiment_name."
        )

    args, config_path = load_args_from_experiment_name(experiments_dir, name_or_tag, wm_orca_args())
    tag = getattr(args, "tag", None) or name_or_tag
    # In YAMLs, training.tag should map to args.tag
    ckpt_dir2 = os.path.join(model_ckpt_root, tag)
    if not os.path.isdir(ckpt_dir2):
        raise FileNotFoundError(
            f"Resolved experiment_name='{name_or_tag}' to tag='{tag}', but checkpoint folder not found: '{ckpt_dir2}'."
        )
    return tag, args, config_path


def select_checkpoints(ckpt_dir: str, max_ckpts: int = 3, max_iteration: int = None) -> List[str]:
    # Expect files like checkpoint-<step>.pt
    pts = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("checkpoint-") and f.endswith(".pt"):
            m = re.search(r"checkpoint-(\d+)\.pt$", f)
            if m:
                step = int(m.group(1))
                pts.append((step, os.path.join(ckpt_dir, f)))
    if not pts:
        return []
    pts.sort(key=lambda x: x[0])
    
    if max_iteration is not None:
        pts = [p for p in pts if p[0] <= max_iteration]
    if len(pts) <= max_ckpts:
        return [p for _, p in pts]
    # pick early, mid, late
    idxs = [0, len(pts)//2, len(pts)-1]
    sel = [pts[i][1] for i in idxs]
    # deduplicate in edge cases
    return list(dict.fromkeys(sel).keys()) if isinstance(sel, dict) else sel


def make_args_for_tag(
    tag: str,
    dataset_root: str,
    dataset_meta_info_path: str = "dataset_meta_info",
) -> wm_orca_args:
    args = wm_orca_args()
    # Heuristic: dataset name often equals the tag prefix up to fps or descriptors
    # If a dataset with the exact tag exists, use that; else try to find a dataset folder that matches a prefix.
    args.dataset_root_path = dataset_root
    datasets_available = set(os.listdir(dataset_root)) if os.path.exists(dataset_root) else set()
    if tag in datasets_available:
        args.dataset_names = tag
    else:
        # try to find a best match (longest common prefix)
        best_match = None
        best_len = -1
        for d in datasets_available:
            common = os.path.commonprefix([tag, d])
            if len(common) > best_len:
                best_len = len(common)
                best_match = d
        if best_match:
            args.dataset_names = best_match
    args.dataset_meta_info_path = dataset_meta_info_path
    args.tag = tag
    args.output_dir = f"model_ckpt/{tag}"
    args.wandb_project_name = "wm_benchmark"
    # Default to 1 view, 256x256 unless specific experiments changed that
    return args


def ensure_uint8(frames: np.ndarray) -> np.ndarray:
    # Expect (T, H, W, 3) float in [-1,1] or [0,1] or uint8; convert to uint8 [0..255]
    if frames.dtype == np.uint8:
        return frames
    f = frames.astype(np.float32)
    if f.max() <= 1.0 and f.min() >= -1.0:
        # map [-1,1] or [0,1] to [0,255]
        f = np.clip((f + 1.0) / 2.0, 0.0, 1.0) if f.min() < 0 else np.clip(f, 0.0, 1.0)
        f = (f * 255.0).round()
    f = np.clip(f, 0.0, 255.0).astype(np.uint8)
    return f


def compute_basic_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    # pred, gt: (T, H, W, 3) uint8
    pred = ensure_uint8(pred)
    gt = ensure_uint8(gt)
    # MSE / MAE averaged over all frames and pixels
    diff = pred.astype(np.float32) - gt.astype(np.float32)
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    # PSNR, SSIM per frame, then average
    psnr_values = []
    ssim_values = []
    if psnr is not None:
        for i in range(pred.shape[0]):
            try:
                psnr_values.append(psnr(gt[i], pred[i], data_range=255))
            except Exception:
                pass
    if ssim is not None:
        for i in range(pred.shape[0]):
            try:
                ssim_values.append(ssim(gt[i], pred[i], channel_axis=2, data_range=255))
            except Exception:
                pass
    return {
        "mse": mse,
        "mae": mae,
        "psnr": float(np.mean(psnr_values)) if psnr_values else float("nan"),
        "ssim": float(np.mean(ssim_values)) if ssim_values else float("nan"),
    }


def resize_mask_to(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # frames: (T, H, W, 3), mask: (T, 1, Hm, Wm) or (T, Hm, Wm)
    # returns boolean mask (T, H, W)
    import torch.nn.functional as F
    t, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    m = mask
    if m.ndim == 3:
        m = m[:, None, :, :]
    # Convert to float tensor for interpolation
    mt = torch.from_numpy(m.astype(np.float32))  # (T, 1, Hm, Wm)
    mt = F.interpolate(mt, size=(h, w), mode="nearest")
    mb = (mt.numpy() > 0.5).astype(bool)
    return mb[:, 0]


def build_roi_masks_from_seg(seg_rgb: np.ndarray, color_threshold: float = 15.0) -> Dict[str, np.ndarray]:
    # seg_rgb: (T, H, W, 3) uint8
    seg = seg_rgb.astype(np.uint8)  # Ensure uint8 dtype
    
    # Extract object color and use closest color matching
    object_color = SEMANTIC_SEGMENTATION_MAPPING["class:object"]
    class_mask_obj = find_closest_color_mask(seg, object_color, color_threshold=color_threshold)
    
    return {
        "object": class_mask_obj,
    }


def apply_mask(frames: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # frames: (T, H, W, 3) uint8, mask: (T, H, W) bool
    # returns (masked_frames, valid_mask) where masked frames are zeroed outside ROI; valid_mask for weighting
    m = mask
    f = frames.copy().astype(np.float32)
    f[~m] = 0.0
    return f.astype(np.uint8), m


class TestDataset:
    """
    Dataset class for test datasets with structure:
    - annotation/*.json
    - videos/{id}/0_rgb.mp4
    - videos/{id}/0_segmentation.mp4
    
    Computes latents on-the-fly using VAE and handles downsampling based on desired FPS.
    """
    def __init__(
        self,
        dataset_root: str,
        args: wm_orca_args,
        mode: str = "val",
        vae: Optional[torch.nn.Module] = None,
        cache_latents: bool = True,
    ):
        self.dataset_root = dataset_root
        self.args = args
        self.mode = mode
        self.vae = vae
        self.cache_latents = cache_latents
        self.horizon_seconds = 1.0

        # Calculate downsampling
        self.original_fps = getattr(args, "original_fps", 50)
        self.desired_fps = args.fps
        self.rgb_skip = max(1, int(self.original_fps / self.desired_fps))
        self.actual_fps = self.original_fps / self.rgb_skip
        
        # Load annotations
        ann_dir = os.path.join(dataset_root, "annotation")
        if not os.path.exists(ann_dir):
            raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
        
        self.annotations = []
        for ann_file in os.listdir(ann_dir):
            if ann_file.endswith(".json"):
                ann_path = os.path.join(ann_dir, ann_file)
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                self.annotations.append(ann)
        
        # Create cache directory for latents
        if self.cache_latents:
            self.cache_dir = os.path.join(dataset_root, "latent_cache", f"{self.desired_fps}fps")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = None
            
        # Compute normalization stats if not already computed
        self._compute_meta_info_if_needed()
    
    def _compute_meta_info_if_needed(self):
        """Load normalization stats from training dataset meta info, not test dataset"""
        # Try to load from args.dataset_meta_info_path (same as training)
        dataset_name = self.args.dataset_names.split('+')[0]  # Use first dataset name
        training_stat_path = os.path.join(self.args.dataset_meta_info_path, dataset_name, "stat.json")
        
        if os.path.exists(training_stat_path):
            # Use training normalization (CORRECT approach)
            with open(training_stat_path, "r", encoding="utf-8") as f:
                stat = json.load(f)
            self.state_p01 = np.array(stat["state_01"])
            self.state_p99 = np.array(stat["state_99"])
            print(f"Using training normalization from: {training_stat_path}")
            return
        
        # Fallback: Check if test dataset has its own stats (for reference, but warn)
        meta_info_dir = os.path.join(self.dataset_root, "meta_info")
        stat_path = os.path.join(meta_info_dir, "stat.json")
        
        if os.path.exists(stat_path):
            print(f"WARNING: Using test dataset normalization instead of training normalization!")
            print(f"This may cause distribution shift. Training stats should be at: {training_stat_path}")
            with open(stat_path, "r", encoding="utf-8") as f:
                stat = json.load(f)
            self.state_p01 = np.array(stat["state_01"])
            self.state_p99 = np.array(stat["state_99"])
            return

        # Compute stats from all annotations
        print(f"Computing meta info for {self.dataset_root}...")
        states_all = []
        for ann in self.annotations:
            if "states" in ann:
                states = np.array(ann["states"])
                # Downsample states to match video frame rate
                states = states[::self.rgb_skip]
                states_all.extend(states.tolist())
        
        if len(states_all) == 0:
            # Fallback: try to extract from observation.state fields
            for ann in self.annotations:
                if "observation.state.cartesian_position" in ann:
                    cart = np.array(ann["observation.state.cartesian_position"])
                    hand = np.array(ann.get("observation.state.hand_joint_position", []))
                    if len(hand) == 0:
                        hand = np.array(ann.get("observation.state.hand_joint_position", []))
                    if len(cart) > 0 and len(hand) > 0:
                        states = np.concatenate([cart, hand], axis=-1)
                        states = states[::self.rgb_skip]
                        states_all.extend(states.tolist())
        
        if len(states_all) == 0:
            # Use default normalization range
            action_dim = self.args.action_dim
            self.state_p01 = np.full(action_dim, -1.0)
            self.state_p99 = np.full(action_dim, 1.0)
            print(f"Warning: Could not compute stats, using default range [-1, 1] for {action_dim} dims")
        else:
            states_arr = np.array(states_all)
            self.state_p01 = np.percentile(states_arr, 1, axis=0)
            self.state_p99 = np.percentile(states_arr, 99, axis=0)
            print(f"Computed stats: state_01 shape={self.state_p01.shape}, state_99 shape={self.state_p99.shape}")
        
        # Save stats
        os.makedirs(meta_info_dir, exist_ok=True)
        stat = {
            "state_01": self.state_p01.tolist(),
            "state_99": self.state_p99.tolist(),
        }
        with open(stat_path, "w", encoding="utf-8") as f:
            json.dump(stat, f, indent=2)
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video using mediapy or decord"""
        if mediapy is not None:
            video = mediapy.read_video(video_path)
            return np.array(video)
        else:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            frames = [vr[i].asnumpy() for i in range(len(vr))]
            return np.array(frames)
    
    def _extract_latents_from_video(
        self, video_path: str, traj_id: str, frame_ids: List[int], device: torch.device
    ) -> torch.Tensor:
        """Extract latents from RGB video, with caching support"""
        cache_path = None
        if self.cache_latents:
            cache_path = os.path.join(self.cache_dir, f"{traj_id}_0.pt")
            if os.path.exists(cache_path):
                # Load cached latents and select frames
                latents = torch.load(cache_path)
                selected = latents[frame_ids]
                return selected
        
        if self.vae is None:
            raise ValueError("VAE must be provided to extract latents from video")
        
        if mediapy is None:
            raise ImportError("mediapy is required for video loading. Install with: pip install mediapy")
        
        # Load video
        video = self._load_video(video_path)
        # Convert to tensor and normalize: (T, H, W, 3) -> (T, 3, H, W), range [0, 255] -> [-1, 1]
        frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
        
        # Downsample temporally
        frames = frames[::self.rgb_skip]
        
        # Resize to target resolution
        target_size = (self.args.height, self.args.width)
        frames = torch.nn.functional.interpolate(
            frames, size=target_size, mode="bilinear", align_corners=False
        )
        
        # Encode to latents using VAE
        vae_model = self.vae.module if hasattr(self.vae, "module") else self.vae
        vae_model.eval()
        
        latents_all = []
        batch_size = 64
        with torch.no_grad():
            frames = frames.to(device)
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]
                latent = vae_model.encode(batch).latent_dist.sample()
                latent = latent.mul_(vae_model.config.scaling_factor)
                latents_all.append(latent.cpu())
        
        latents = torch.cat(latents_all, dim=0)  # (T_downsampled, 4, Hc, Wc)
        
        # Cache if enabled
        if self.cache_latents and cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(latents, cache_path)
        
        # Select requested frames (adjust frame_ids for downsampled sequence)
        frame_ids_downsampled = [fid // self.rgb_skip for fid in frame_ids]
        frame_ids_downsampled = [min(fid, len(latents) - 1) for fid in frame_ids_downsampled]
        selected = latents[frame_ids_downsampled]
        
        return selected
    
    def _load_segmentation_video(self, seg_path: str, frame_ids: List[int]) -> Optional[np.ndarray]:
        """Load segmentation video and extract RGB frames"""
        if not os.path.exists(seg_path):
            return None
        
        try:
            video = self._load_video(seg_path)
            print(f"Segmentation video shape: {video.shape}")
            # Downsample temporally
            video = video[::self.rgb_skip]
            # Select frames (adjust for downsampling)
            frame_ids_downsampled = [min(fid // self.rgb_skip, len(video) - 1) for fid in frame_ids]
            selected = video[frame_ids_downsampled]
            # Ensure RGB (3 channels) - take only first 3 channels if more exist
            if selected.shape[-1] > 3:
                selected = selected[..., :3]
            elif selected.shape[-1] < 3:
                raise ValueError(f"Segmentation video has {selected.shape[-1]} channels, expected 3")
            return selected.astype(np.uint8)
        except Exception as e:
            print(f"Warning: Could not load segmentation video {seg_path}: {e}")
            return None
    
    def normalize_bound(
        self,
        data: np.ndarray,
        state_p01: Optional[np.ndarray] = None,
        state_p99: Optional[np.ndarray] = None,
        clip_min: float = -1,
        clip_max: float = 1,
    ) -> np.ndarray:
        """Normalize state data using computed percentiles"""
        eps = 1e-8
        p01 = state_p01 if state_p01 is not None else self.state_p01
        p99 = state_p99 if state_p99 is not None else self.state_p99
        
        # Ensure broadcasting works: data shape (..., dim) and p01/p99 shape (dim,)
        ndata = 2 * (data - p01) / (p99 - p01 + eps) - 1
        return np.clip(ndata, clip_min, clip_max)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset"""
        ann = self.annotations[idx]
        traj_id = ann.get("episode_id", str(idx))
        
        # Extract states and prepare actions (handle both "states" key and separate observation keys)
        cart = np.array(ann["observation.state.cartesian_pose"])
        hand = np.array(ann.get("observation.state.hand_joint_position", []))
        states = np.concatenate([cart, hand], axis=-1)
        
        # Downsample states
        states = states[::self.rgb_skip]
        
        # Determine frame sequence (history + current + future)
        # We need num_history frames before current, then current, then num_frames for future
        # num_frames is the number of future frames to predict
        # num_history is the number of history frames to use for prediction
        num_total_frames = self.args.num_history + self.args.fps * self.horizon_seconds
        
        # Select a valid starting point (need at least num_history frames before)
        max_start = max(0, len(states) - num_total_frames)
        # For validation, use middle of available range for more variety
        if self.mode == "train":
            start_idx = 0
        else:
            # Use a deterministic start based on trajectory ID for reproducibility
            start_idx = hash(traj_id) % max(1, max_start + 1) if max_start > 0 else 0
        
        # Select frame indices: [history frames..., current, future frames...]
        frame_ids = []
        current_idx = start_idx + self.args.num_history
        
        # History frames (before current)
        for i in range(self.args.num_history):
            hist_idx = current_idx - (self.args.num_history - i)
            frame_ids.append(max(0, min(hist_idx, len(states) - 1)))
        
        # Current frame
        frame_ids.append(min(current_idx, len(states) - 1))
        
        # Future frames
        for i in range(1, self.args.num_frames):
            future_idx = current_idx + i
            frame_ids.append(min(future_idx, len(states) - 1))
        
        # Ensure we have exactly the expected number of frames
        while len(frame_ids) < num_total_frames:
            frame_ids.append(frame_ids[-1])
        frame_ids = frame_ids[:num_total_frames]
        
        # Extract corresponding states for actions
        state_selected = states[frame_ids]  # (num_total_frames, state_dim)
        state_dim = state_selected.shape[-1]
        
        # Handle different action space configurations (match Dataset_mix logic)
        if self.args.use_only_hand_actions:
            # Expect states to have cartesian (6) + hand joints
            if state_dim >= 23:
                hand_start = 6
                hand_data = state_selected[:, hand_start:]
                if self.args.use_average_scalar_hand_action:
                    action = np.mean(hand_data, axis=-1, keepdims=True)
                else:
                    action = hand_data
                # Normalize using corresponding stats slice
                hand_dim = action.shape[-1]
                p01_slice = self.state_p01[hand_start:hand_start+hand_dim] if len(self.state_p01) > hand_start else self.state_p01
                p99_slice = self.state_p99[hand_start:hand_start+hand_dim] if len(self.state_p99) > hand_start else self.state_p99
                action = self.normalize_bound(action, state_p01=p01_slice, state_p99=p99_slice)
            else:
                # Fallback: use all available dimensions, pad/truncate to match action_dim
                min_dim = min(state_dim, len(self.state_p01))
                action = self.normalize_bound(
                    state_selected[:, :min_dim],
                    state_p01=self.state_p01[:min_dim],
                    state_p99=self.state_p99[:min_dim]
                )
                if action.shape[-1] < self.args.action_dim:
                    padding = np.zeros((action.shape[0], self.args.action_dim - action.shape[-1]))
                    action = np.concatenate([action, padding], axis=-1)
                elif action.shape[-1] > self.args.action_dim:
                    action = action[:, :self.args.action_dim]
        elif self.args.use_only_ee_pose_actions:
            # Use only first 6 dimensions (cartesian pose)
            if state_dim >= 6:
                cart_data = state_selected[:, :6]
                min_dim = min(6, len(self.state_p01))
                action = self.normalize_bound(
                    cart_data[:, :min_dim],
                    state_p01=self.state_p01[:min_dim],
                    state_p99=self.state_p99[:min_dim]
                )
                if action.shape[-1] < self.args.action_dim:
                    padding = np.zeros((action.shape[0], self.args.action_dim - action.shape[-1]))
                    action = np.concatenate([action, padding], axis=-1)
                elif action.shape[-1] > self.args.action_dim:
                    action = action[:, :self.args.action_dim]
            else:
                # Fallback
                min_dim = min(state_dim, len(self.state_p01))
                action = self.normalize_bound(
                    state_selected[:, :min_dim],
                    state_p01=self.state_p01[:min_dim],
                    state_p99=self.state_p99[:min_dim]
                )
                if action.shape[-1] < self.args.action_dim:
                    padding = np.zeros((action.shape[0], self.args.action_dim - action.shape[-1]))
                    action = np.concatenate([action, padding], axis=-1)
                elif action.shape[-1] > self.args.action_dim:
                    action = action[:, :self.args.action_dim]
        else:
            # Use full state (cartesian + hand joints) - default
            # Handle dimension mismatch between state_dim and normalization stats
            if state_dim == len(self.state_p01):
                action = self.normalize_bound(state_selected)
            else:
                # Dimension mismatch: use minimum available dimension
                min_dim = min(state_dim, len(self.state_p01))
                action = self.normalize_bound(
                    state_selected[:, :min_dim],
                    state_p01=self.state_p01[:min_dim],
                    state_p99=self.state_p99[:min_dim]
                )
                # Pad or truncate to match expected action_dim
                if action.shape[-1] < self.args.action_dim:
                    # Pad with zeros
                    padding = np.zeros((action.shape[0], self.args.action_dim - action.shape[-1]))
                    action = np.concatenate([action, padding], axis=-1)
                elif action.shape[-1] > self.args.action_dim:
                    # Truncate
                    action = action[:, :self.args.action_dim]
        
        # Load RGB video and extract latents
        video_path = os.path.join(self.dataset_root, "videos", str(traj_id), "0_rgb.mp4")
        device = next(self.vae.parameters()).device if self.vae else torch.device("cpu")
        latents = self._extract_latents_from_video(video_path, traj_id, frame_ids, device)
        
        # Load segmentation if available
        seg_path = os.path.join(self.dataset_root, "videos", str(traj_id), "0_segmentation.mp4")
        seg_rgb = self._load_segmentation_video(seg_path, frame_ids)
        
        # Prepare data dict
        data = {
            "latent": latents.float(),  # (F, 4, Hc, Wc)
            "action": torch.tensor(action).float(),  # (F, action_dim)
            "text": ann.get("texts", [""])[0] if "texts" in ann else "",
            "episode_id": traj_id,
        }
        
        if seg_rgb is not None:
            data["seg_rgb"] = seg_rgb
        
        return data


def calculate_pixel_error(gt_frames: np.ndarray, pred_frames: np.ndarray, error_type: str = 'mse') -> np.ndarray:
    """
    Calculate pixel-wise error between ground truth and predictions.
    
    Args:
        gt_frames: Ground truth frames (T, H, W, 3) uint8
        pred_frames: Predicted frames (T, H, W, 3) uint8
        error_type: Type of error metric ('mse' or 'mae')
        
    Returns:
        errors: Array of shape (T, H, W) with error values
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
        errors: Array of shape (T, H, W) with error values
        output_path: Path to save the animation
        fps: Frames per second for the animation
        cmap: Colormap to use for heatmap
        interval: Delay between frames in milliseconds
        vmin: Minimum value for colormap (None for auto)
        vmax: Maximum value for colormap (None for auto)
    """
    if animation is None:
        print("Warning: matplotlib.animation not available, skipping heatmap animation")
        return
    
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
    output_path_obj = os.path.splitext(output_path)[0]  # Remove extension to add .mp4
    anim_path = f"{output_path_obj}_heatmap_evolution.mp4"
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000) if animation else None
    if writer:
        anim.save(anim_path, writer=writer)
        print(f"Heatmap animation saved to: {anim_path}")
    
    plt.close()


def plot_error_statistics(errors: np.ndarray, output_path: str):
    """
    Plot error statistics over time.
    
    Args:
        errors: Array of shape (T, H, W) with error values
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
    output_path_obj = os.path.splitext(output_path)[0]
    stats_path = f"{output_path_obj}_error_statistics.png"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error statistics plot saved to: {stats_path}")


def plot_spatial_error_map(errors: np.ndarray, output_path: str):
    """
    Plot average spatial error map across all frames.
    
    Args:
        errors: Array of shape (T, H, W) with error values
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
    output_path_obj = os.path.splitext(output_path)[0]
    spatial_path = f"{output_path_obj}_spatial_error.png"
    plt.savefig(spatial_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spatial error map saved to: {spatial_path}")


def average_metric_with_mask(metric_name: str, pred: np.ndarray, gt: np.ndarray, m: np.ndarray) -> float:
    # pred, gt: (T, H, W, 3) uint8; m: (T, H, W) bool
    # For MSE/MAE compute over ROI pixels only; For PSNR/SSIM average per-frame computed on masked frames
    if metric_name in ("mse", "mae"):
        diff = pred.astype(np.float32) - gt.astype(np.float32)
        if metric_name == "mse":
            val = np.sum((diff**2) * m[..., None]) / (np.sum(m) * 3 + 1e-8)
        else:
            val = np.sum(np.abs(diff) * m[..., None]) / (np.sum(m) * 3 + 1e-8)
        return float(val)
    elif metric_name in ("psnr", "ssim"):
        if psnr is None or ssim is None:
            return float("nan")
        vals = []
        for i in range(pred.shape[0]):
            if not np.any(m[i]):
                continue
            p_masked = pred[i].copy()
            g_masked = gt[i].copy()
            p_masked[~m[i]] = 0
            g_masked[~m[i]] = 0
            if metric_name == "psnr":
                vals.append(psnr(g_masked, p_masked, data_range=255))
            else:
                vals.append(ssim(g_masked, p_masked, channel_axis=2, data_range=255))
        return float(np.mean(vals)) if vals else float("nan")
    else:
        return float("nan")


def evaluate_one_batch(
    model: CrtlWorld,
    accelerator: Accelerator,
    batch: Dict,
    args: wm_orca_args,
    horizon_seconds: int,
    seg_rgb: Optional[np.ndarray] = None,
    save_sample_path: Optional[str] = None,
) -> Dict[str, float]:
    # Prepare current frame and history latents and actions
    device = accelerator.device
    pipeline = model.module.pipeline if accelerator.num_processes > 1 else model.pipeline

    video_gt = batch["latent"].to(device, non_blocking=True)  # (F, 4, Hc, Wc) or (B, F, 4, Hc, Wc)
    if video_gt.dim() == 4:
        video_gt = video_gt.unsqueeze(0)
    actions = batch["action"].to(device, non_blocking=True)   # (B, F, action_dim)
    texts = batch["text"]
    his_latent_gt, future_latent_ft = video_gt[:, :args.num_history], video_gt[:, args.num_history:]
    current_latent = future_latent_ft[:, 0]

    # Build action latent
    with torch.no_grad():
        action_latent = (
            model.module.action_encoder(actions, texts, model.module.tokenizer, model.module.text_encoder, args.frame_level_cond)
            if accelerator.num_processes > 1
            else model.action_encoder(actions, texts, model.tokenizer, model.text_encoder, args.frame_level_cond)
        )

    # Determine num_frames for horizon
    num_frames = int(args.fps * horizon_seconds)
    num_frames = max(1, num_frames)
    print(f"num_frames: {num_frames}")
    print(f"his_latent_gt shape: {his_latent_gt.shape}")
    print(f"future_latent_ft shape: {future_latent_ft.shape}")
    print(f"current_latent shape: {current_latent.shape}")
    print(f"action_latent shape: {action_latent.shape}")

    # Run pipeline to predict latents
    with torch.no_grad():
        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(args.num_views * args.height),
            num_frames=num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )

    print(f"pred_latents shape: {pred_latents.shape}")
    print(f"future_latent_ft shape: {future_latent_ft.shape}")

    # Prepare GT and predicted latent sequences for future window
    # Assemble GT future latents for the same time span (clamp by availability)
    pred_latents = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=args.num_views,n=1) # (B, 8, 4, 32,32)
    gt_latents = torch.cat([his_latent_gt, future_latent_ft], dim=1) # (B, 8, 4, 32,32)
    gt_latents = einops.rearrange(gt_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=args.num_views, n=1) # (B, 8, 4, 32,32)

    # Decode both to RGB frames
    def decode_latents(latents: torch.Tensor) -> np.ndarray:
        bsz, frame_num = latents.shape[:2]
        lat = latents.flatten(0, 1)
        decoded = []
        decode_kwargs = {}
        for i in range(0, lat.shape[0], args.decode_chunk_size):
            chunk = lat[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        frames = torch.cat(decoded, dim=0)
        frames = frames.reshape(bsz, frame_num, *frames.shape[1:])
        frames = ((frames / 2.0 + 0.5).clamp(0, 1) * 255).to(pipeline.unet.dtype).detach().cpu().numpy()
        # to uint8 (B, T, C, H, W) -> (T, H, W, C)
        frames = frames.astype(np.uint8).transpose(0, 1, 3, 4, 2)[0]
        return frames

    gt_frames = decode_latents(gt_latents)
    pred_frames = decode_latents(pred_latents)

    # Align frames for comparison (future frames only)
    gt_future = gt_frames[args.num_history:args.num_history + pred_frames.shape[0]]
    min_len = min(gt_future.shape[0], pred_frames.shape[0])
    gt_future = gt_future[:min_len]
    pred_frames_aligned = pred_frames[:min_len]

    # Save visualization video if requested
    if save_sample_path is not None and mediapy is not None:
        # Stack GT (top) and Pred (bottom)
        combined = np.concatenate([gt_future, pred_frames_aligned], axis=1)
        os.makedirs(os.path.dirname(save_sample_path), exist_ok=True)
        mediapy.write_video(save_sample_path, combined, fps=args.fps)
        
        # Generate error heatmaps and statistics for this sample
        print(f"Generating error heatmaps for sample...")
        
        # Calculate pixel errors (MSE and MAE)
        errors_mse = calculate_pixel_error(gt_future, pred_frames_aligned, error_type='mse')
        errors_mae = calculate_pixel_error(gt_future, pred_frames_aligned, error_type='mae')
        
        # Generate heatmap evolution animations
        base_path = os.path.splitext(save_sample_path)[0]
        
        # MSE heatmap evolution
        plot_heatmap_evolution(errors_mse, base_path, fps=args.fps, cmap='hot', interval=100)
        
        # MAE heatmap evolution
        mae_base = f"{base_path}_mae"
        plot_heatmap_evolution(errors_mae, mae_base, fps=args.fps, cmap='hot', interval=100)
        
        # Error statistics plots
        plot_error_statistics(errors_mse, base_path)
        
        # Spatial error maps (MSE and MAE)
        plot_spatial_error_map(errors_mse, base_path)
        plot_spatial_error_map(errors_mae, mae_base)

    # Basic metrics (whole frame)
    # we only compare the future frames since the history frames are not predicted
    base = compute_basic_metrics(pred_frames, gt_frames[args.num_history:])

    # ROI masks if segmentation RGB available
    roi_results: Dict[str, float] = {}
    if seg_rgb is not None and seg_rgb.shape[0] >= gt_frames.shape[0]:
        # Align time window
        seg_t = seg_rgb[: gt_frames.shape[0]]
        
        # Debug: Print unique colors in first frame to verify segmentation format
        if seg_t.shape[0] > 0:
            unique_colors = np.unique(seg_t[0].reshape(-1, 3), axis=0)
            print(f"Debug: Found {len(unique_colors)} unique colors in first segmentation frame")
            print(f"Debug: Sample colors (first 5): {unique_colors[:5]}")
            print(f"Debug: Expected object color: {SEMANTIC_SEGMENTATION_MAPPING['class:object']}")
        
        # Ensure seg_t is uint8
        seg_t = seg_t.astype(np.uint8)
        
        # Build object mask using closest color matching
        color_threshold = 15.0  # Euclidean distance threshold in RGB space
        roi_masks = build_roi_masks_from_seg(seg_t, color_threshold=color_threshold)
        object_mask = roi_masks["object"]
        
        # Hand = not in mapping classes; define as not object and not robot/ground/table
        # Use closest color matching to handle video compression artifacts
        all_known = np.zeros_like(object_mask, dtype=bool)
        for class_name, color in SEMANTIC_SEGMENTATION_MAPPING.items():
            if "hand" not in class_name:  # Skip hand since it's derived from what's not in mapping
                mask = find_closest_color_mask(seg_t, color, color_threshold=color_threshold)
                all_known |= mask
        
        hand_mask = ~all_known
        background_mask = ~(object_mask | hand_mask)
        
        # Debug: Print mask statistics
        print(f"Debug: object_mask pixels: {np.sum(object_mask)} / {object_mask.size} ({100*np.mean(object_mask):.1f}%)")
        print(f"Debug: hand_mask pixels: {np.sum(hand_mask)} / {hand_mask.size} ({100*np.mean(hand_mask):.1f}%)")
        print(f"Debug: background_mask pixels: {np.sum(background_mask)} / {background_mask.size} ({100*np.mean(background_mask):.1f}%)")

        # save an image of the masks - FIXED: convert to uint8 for proper visualization
        # Convert boolean masks to uint8 (0 or 255) for proper visualization
        hand_vis = (hand_mask[0].astype(np.uint8) * 255)
        object_vis = (object_mask[0].astype(np.uint8) * 255)
        background_vis = (background_mask[0].astype(np.uint8) * 255)
        
        # Concatenate along width (axis=1) to place masks side by side
        mask_image = np.concatenate([hand_vis, object_vis, background_vis], axis=1)  # axis=1 is width for 2D array
        
        # Create a figure with subplots for better visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(hand_vis, cmap='gray')
        axes[0].set_title('Hand Mask')
        axes[0].axis('off')
        
        axes[1].imshow(object_vis, cmap='gray')
        axes[1].set_title('Object Mask')
        axes[1].axis('off')
        
        axes[2].imshow(background_vis, cmap='gray')
        axes[2].set_title('Background Mask')
        axes[2].axis('off')
        
        axes[3].imshow(mask_image, cmap='gray')
        axes[3].set_title('All Masks Combined')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(save_sample_path), "masks.png"), dpi=150, bbox_inches='tight')
        plt.close()

        for name, mask in [("hand", hand_mask), ("object", object_mask), ("background", background_mask)]:
            for k in ("mse", "mae", "psnr", "ssim"):
                val = average_metric_with_mask(k, pred_frames, gt_frames[args.num_history:], mask[args.num_history:])
                roi_results[f"{name}_{k}"] = val
    else:
        # If segmentation not available, skip ROI metrics
        pass

    return {**base, **roi_results}


def evaluate_dataset_for_checkpoint(
    tag: str,
    args: wm_orca_args,
    ckpt_path: str,
    dataset_root: str,
    analysis_root: str,
    horizons_s: List[int],
    max_batches: Optional[int] = None,
    use_test_dataset: bool = False,
) -> Dict:
    # Start from the config used in training if we can resolve it later; for now keep defaults.
    args.ckpt_path = ckpt_path
    os.makedirs(analysis_root, exist_ok=True)
    save_dir = os.path.join(analysis_root, tag, os.path.basename(ckpt_path).replace(".pt", ""))
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator(log_with=None)
    
    # Load VAE for test dataset if needed
    vae = None
    if use_test_dataset:
        if AutoencoderKLTemporalDecoder is None:
            raise ImportError("AutoencoderKLTemporalDecoder required for test dataset")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_model_path, subfolder="vae")
        vae = accelerator.prepare(vae)
        vae.eval()
    
    model = CrtlWorld(args)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(accelerator.device)
    model.eval()

    # Choose dataset class based on structure
    if use_test_dataset:
        val_dataset = TestDataset(dataset_root, args, mode="val", vae=vae, cache_latents=True)
    else:
        val_dataset = Dataset_mix(args, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    metrics_agg: Dict[str, Dict[int, List[float]]] = {}
    vram_stats: Dict[int, List[int]] = {}

    for horizon in horizons_s:
        metrics_agg[horizon] = {}
        vram_stats[horizon] = []

        with torch.no_grad():
            # set the horizon_seconds for the dataset
            val_dataset.horizon_seconds = horizon

            for i, batch in enumerate(val_loader):
                if max_batches is not None and i >= max_batches:
                    break

                # Extract segmentation RGB from batch if available (from TestDataset)
                seg_rgb = None
                if "seg_rgb" in batch:
                    seg_data = batch["seg_rgb"]
                    if isinstance(seg_data, (list, tuple)) and len(seg_data) > 0:
                        seg_rgb = seg_data[0] if isinstance(seg_data[0], np.ndarray) else seg_data[0].numpy()
                    elif isinstance(seg_data, torch.Tensor):
                        seg_rgb = seg_data[0].numpy() if seg_data.dim() > 1 else seg_data.numpy()
                    elif isinstance(seg_data, np.ndarray):
                        seg_rgb = seg_data[0] if seg_data.ndim > 1 else seg_data

                print(f"Evaluating horizon: {horizon}s")
                # Adjust args for horizon and reset peak memory before call
                args.num_frames = int(args.fps * horizon)
                args.num_frames = max(1, args.num_frames)
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                start = time.time()

                sample_path = os.path.join(save_dir, "samples", f"h{horizon}s_sample{i}.mp4")
                out = evaluate_one_batch(
                    model,
                    accelerator,
                    batch,
                    args,
                    horizon_seconds=horizon,
                    seg_rgb=seg_rgb,
                    save_sample_path=sample_path,
                )

                # aggregate
                for k, v in out.items():
                    metrics_agg[horizon].setdefault(k, []).append(v)

                # record VRAM
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated()
                    vram_stats[horizon].append(int(peak))

                _ = time.time() - start

    # reduce to means
    summary = {}
    for horizon, kv in metrics_agg.items():
        summary[horizon] = {k: float(np.nanmean(vs)) if len(vs) > 0 else float("nan") for k, vs in kv.items()}
        if vram_stats[horizon]:
            summary[horizon]["peak_vram_bytes_mean"] = float(np.mean(vram_stats[horizon]))
            summary[horizon]["peak_vram_bytes_max"] = float(np.max(vram_stats[horizon]))

    # save summary
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump({"tag": tag, "ckpt": ckpt_path, "metrics": summary}, f, indent=2)

    return {"tag": tag, "ckpt": ckpt_path, "metrics": summary}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt_root", type=str, default="model_ckpt")
    parser.add_argument("--dataset_root", type=str, default="test_datasets")
    parser.add_argument("--analysis_root", type=str, default=f"test_analysis_wm_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--max_ckpt_iteration", type=int, default=80000)
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--max_ckpts_per_tag", type=int, default=3)
    parser.add_argument("--max_batches", type=int, default=None, help="Limit for quick runs; None = full dataset")
    parser.add_argument("--short_horizon_s", type=int, default=1)
    parser.add_argument("--long_horizon_s", type=int, default=2)
    args_cli = parser.parse_args()

    groups = {
        "action_encoder_size_test": {
            "test_tags": [
                "action_encoder_size_test_1024_sine",
                # "action_encoder_size_test_2x2048_sine",
                # "action_encoder_size_test_4x1024_sine",
                # "action_encoder_size_test_1024_random",
                # "action_encoder_size_test_2x1024_random",
                # "action_encoder_size_test_4x4096_random",
            ],
            "test_dataset_path": "/data/faive_lab/datasets/converted_to_lerobot/2026-01-08T09-53-35/fixed_ee_moving_fingers_lowres_test_dataset",
        },
        # "hand_weighting_test": {
        #     "test_tags": [
        #         "hand_weighting_test_1.0",
        #         "hand_weighting_test_2.0",
        #         "hand_weighting_test_3.0",
        #     ],
        #     "test_dataset_path": "datasets/2025-12-24T01-20-32",
        # },
        # "ee_vs_finger_test": {
        #     "test_tags": [
        #         "ee_vs_finger_test_fixed_ee",
        #         "ee_vs_finger_test_relative_ee",
        #     ],
        #     "test_dataset_path": "datasets/2025-12-24T01-20-32",
        # },
        # "finger_motion_type_test": {
        #     "test_tags": [
        #         "finger_motion_type_test_sine",
        #         "finger_motion_type_test_random",
        #         "finger_motion_type_test_human",
        #     ],
        #     "test_dataset_path": "datasets/2025-12-24T01-20-32",
        # },
        # "open_close_scalar_vs_full": {
        #     "test_tags": [
        #         "open_close_scalar_vs_full_scalar",
        #         "open_close_scalar_vs_full_all_finger",
        #     ],
        #     "test_dataset_path": "datasets/2025-12-24T01-20-32",
        # },
    }

    all_results: Dict[str, List[Dict]] = {}
    for group_name, group_tags in groups.items():
        group_results = []
        test_dataset_path = group_tags["test_dataset_path"]
        for tag in group_tags["test_tags"]:
            resolved_tag, resolved_args, resolved_yaml = resolve_ckpt_tag_and_args(
                name_or_tag=tag,
                model_ckpt_root=args_cli.model_ckpt_root,
                dataset_root=test_dataset_path,
                experiments_dir=args_cli.experiments_dir,
            )
            ckpt_dir = os.path.join(args_cli.model_ckpt_root, resolved_tag)
            ckpts = select_checkpoints(
                ckpt_dir,
                max_ckpts=args_cli.max_ckpts_per_tag,
                max_iteration=args_cli.max_ckpt_iteration,
            )
            for ckpt in ckpts:
                print(
                    f"[{group_name}] Evaluating name='{tag}' -> tag='{resolved_tag}' ckpt='{os.path.basename(ckpt)}' ...",
                    flush=True,
                )
                # Detect if dataset_root is a test dataset (has annotation/ and videos/ folders)
                use_test_dataset = (
                    os.path.exists(os.path.join(test_dataset_path, "annotation")) and
                    os.path.exists(os.path.join(test_dataset_path, "videos"))
                )
                
                res = evaluate_dataset_for_checkpoint(
                    tag=resolved_tag,
                    args=resolved_args,
                    ckpt_path=ckpt,
                    dataset_root=test_dataset_path,
                    analysis_root=args_cli.analysis_root,
                    horizons_s=[args_cli.short_horizon_s, args_cli.long_horizon_s],
                    max_batches=args_cli.max_batches,
                    use_test_dataset=use_test_dataset,
                )
                res["experiment_name"] = tag
                res["resolved_tag"] = resolved_tag
                res["experiment_yaml"] = resolved_yaml
                res["group"] = group_name
                group_results.append(res)
        all_results[group_name] = group_results

    # Save global summary and generate comparison plots
    os.makedirs(args_cli.analysis_root, exist_ok=True)
    overall_path = os.path.join(args_cli.analysis_root, "overall_summary.json")
    with open(overall_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Flatten results for plotting
    rows = []
    for group_name, group_results in all_results.items():
        for res in group_results:
            tag = res.get("resolved_tag", "")
            ckpt = os.path.basename(res.get("ckpt", ""))
            metrics = res.get("metrics", {})
            for horizon, metric_dict in metrics.items():
                for k, v in metric_dict.items():
                    rows.append({
                        "group": group_name,
                        "tag": tag,
                        "checkpoint": ckpt,
                        "horizon_s": horizon,
                        "metric": k,
                        "value": v,
                    })
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args_cli.analysis_root, "metrics_summary.csv")
        df.to_csv(csv_path, index=False)

        # Plot per-horizon, per-metric comparisons across tags/checkpoints
        metrics_to_plot = sorted(df["metric"].unique())
        for horizon in sorted(df["horizon_s"].unique()):
            df_h = df[df["horizon_s"] == horizon]
            for metric in metrics_to_plot:
                df_m = df_h[df_h["metric"] == metric]
                if df_m.empty:
                    continue
                plt.figure(figsize=(12, 6))
                labels = df_m.apply(lambda r: f"{r['tag']}\\n{r['checkpoint']}", axis=1)
                plt.bar(labels, df_m["value"])
                plt.title(f"{metric} @ {horizon}s")
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plot_dir = os.path.join(args_cli.analysis_root, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(os.path.join(plot_dir, f"{metric}_h{horizon}s.png"))
                plt.close()

    print(f"Saved analysis under: {args_cli.analysis_root}")


if __name__ == "__main__":
    main()