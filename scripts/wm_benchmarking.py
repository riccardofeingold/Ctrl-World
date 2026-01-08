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
from torch.utils.data import DataLoader

from accelerate import Accelerator

from config import wm_orca_args
from utils.config_loader import load_experiment_config
from dataset.dataset_orca import Dataset_mix
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

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

# =============================================================================
# Configuration for segmentation classes (pixel colors as RGBA)
# Hand is defined as "everything not in these classes"
# =============================================================================
SEMANTIC_SEGMENTATION_MAPPING: Dict[str, Tuple[int, int, int, int]] = {
    "class:object": (0, 0, 255, 255),
    "class:robot": (0, 255, 0, 255),
    "class:ground": (255, 0, 0, 255),
    "class:table": (255, 255, 0, 255),
}


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


def build_roi_masks_from_seg(seg_rgba: np.ndarray) -> Dict[str, np.ndarray]:
    # seg_rgba: (T, H, W, 4) uint8
    seg = seg_rgba
    h, w = seg.shape[1], seg.shape[2]
    Tn = seg.shape[0]
    # Flatten to compare colors
    flat = seg.reshape(Tn, h * w, 4)
    obj_color = np.array(SEMANTIC_SEGMENTATION_MAPPING["class:object"], dtype=np.uint8)[None, None, :]
    class_mask_obj = np.all(flat == obj_color, axis=-1).reshape(Tn, h, w)
    # background: not object and not hand; we define hand later; for now background = not object, will be refined by caller
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
    seg_rgba: Optional[np.ndarray] = None,
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

    # Prepare GT and predicted latent sequences for future window
    # Assemble GT future latents for the same time span (clamp by availability)
    future_len = min(num_frames, future_latent_ft.shape[1])
    gt_latents = future_latent_ft[:, :future_len]
    pred_latents = pred_latents[:, :future_len]

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

    # Basic metrics (whole frame)
    base = compute_basic_metrics(pred_frames, gt_frames)

    # ROI masks if segmentation RGBA available
    roi_results: Dict[str, float] = {}
    if seg_rgba is not None and seg_rgba.shape[0] >= gt_frames.shape[0]:
        # Align time window
        seg_t = seg_rgba[: gt_frames.shape[0]]
        # Build object mask
        roi_masks = build_roi_masks_from_seg(seg_t)
        object_mask = roi_masks["object"]
        # Hand = not in mapping classes; define as not object and not robot/ground/table
        all_known = np.zeros_like(object_mask)
        for _, color in SEMANTIC_SEGMENTATION_MAPPING.items():
            c = np.array(color, dtype=np.uint8)[None, None, :]
            all_known |= np.all(seg_t == c, axis=-1)
        hand_mask = ~all_known
        background_mask = ~(object_mask | hand_mask)

        for name, mask in [("hand", hand_mask), ("object", object_mask), ("background", background_mask)]:
            for k in ("mse", "mae", "psnr", "ssim"):
                val = average_metric_with_mask(k, pred_frames, gt_frames, mask)
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
) -> Dict:
    # Start from the config used in training if we can resolve it later; for now keep defaults.
    args.ckpt_path = ckpt_path
    os.makedirs(analysis_root, exist_ok=True)
    save_dir = os.path.join(analysis_root, tag, os.path.basename(ckpt_path).replace(".pt", ""))
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator(log_with=None)
    model = CrtlWorld(args)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(accelerator.device)
    model.eval()

    val_dataset = Dataset_mix(args, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    metrics_agg: Dict[str, Dict[int, List[float]]] = {}
    vram_stats: Dict[int, List[int]] = {}

    for horizon in horizons_s:
        metrics_agg[horizon] = {}
        vram_stats[horizon] = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break

            # Try to load segmentation RGBA frames if present:
            seg_rgba = None
            try:
                # If you keep segmentation videos in resized dataset, adjust here:
                # Not guaranteed; left as optional.
                seg_rgba = None
            except Exception:
                seg_rgba = None

            for horizon in horizons_s:
                # Adjust args for horizon and reset peak memory before call
                args.num_frames = int(args.fps * horizon)
                args.num_frames = max(1, args.num_frames)
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                start = time.time()

                out = evaluate_one_batch(model, accelerator, batch, args, horizon_seconds=horizon, seg_rgba=seg_rgba)

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
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt_root", type=str, default="model_ckpt")
    parser.add_argument("--dataset_root", type=str, default="test_datasets")
    parser.add_argument("--analysis_root", type=str, default=f"test_analysis_wm_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--max_ckpt_iteration", type=int, default=80000)
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--max_ckpts_per_tag", type=int, default=3)
    parser.add_argument("--max_batches", type=int, default=None, help="Limit for quick runs; None = full dataset")
    parser.add_argument("--short_horizon_s", type=int, default=1)
    parser.add_argument("--long_horizon_s", type=int, default=5)
    args_cli = parser.parse_args()

    groups = {
        "action_encoder_size_test": {
            "test_tags": [
                "action_encoder_size_test_1024_sine",
                "action_encoder_size_test_2x2048_sine",
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
                res = evaluate_dataset_for_checkpoint(
                    tag=resolved_tag,
                    args=resolved_args,
                    ckpt_path=ckpt,
                    dataset_root=args_cli.dataset_root,
                    analysis_root=args_cli.analysis_root,
                    horizons_s=[args_cli.short_horizon_s, args_cli.long_horizon_s],
                    max_batches=args_cli.max_batches,
                )
                res["experiment_name"] = tag
                res["resolved_tag"] = resolved_tag
                res["experiment_yaml"] = resolved_yaml
                res["group"] = group_name
                group_results.append(res)
        all_results[group_name] = group_results

    # Save global summary
    os.makedirs(args_cli.analysis_root, exist_ok=True)
    with open(os.path.join(args_cli.analysis_root, "overall_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved analysis under: {args_cli.analysis_root}")


if __name__ == "__main__":
    main()