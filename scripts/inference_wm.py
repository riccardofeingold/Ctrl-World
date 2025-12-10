# from diffusers import StableVideoDiffusionPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.ctrl_world import CrtlWorld
from dataset.dataset_orca import Dataset_mix

import numpy as np
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import wandb
import swanlab
import mediapy
from models.ctrl_world import CrtlWorld
from config import wm_orca_args
import math
import copy


def compute_video_metrics(pred_frames, gt_frames):
    """
    Compute multiple metrics to evaluate predicted video quality against ground truth.
    
    Args:
        pred_frames: (B, T, H, W, 3) predicted frames in range [0, 255], uint8
        gt_frames: (B, T, H, W, 3) ground truth frames in range [0, 255], uint8
    
    Returns:
        dict with metrics: psnr, ssim, mse, mae, lpips
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Flatten batch and time dimensions
    pred_flat = pred_frames.reshape(-1, *pred_frames.shape[2:])  # (B*T, H, W, 3)
    gt_flat = gt_frames.reshape(-1, *gt_frames.shape[2:])  # (B*T, H, W, 3)
    
    # Convert to float for calculations
    pred_float = pred_flat.astype(np.float32)
    gt_float = gt_flat.astype(np.float32)
    
    # 1. MSE (Mean Squared Error) - lower is better
    mse = np.mean((pred_float - gt_float) ** 2)
    
    # 2. MAE (Mean Absolute Error) - lower is better
    mae = np.mean(np.abs(pred_float - gt_float))
    
    # 3. PSNR (Peak Signal-to-Noise Ratio) - higher is better
    psnr_values = []
    for i in range(len(pred_flat)):
        psnr_val = psnr(gt_flat[i], pred_flat[i], data_range=255)
        psnr_values.append(psnr_val)
    avg_psnr = np.mean(psnr_values)
    
    # 4. SSIM (Structural Similarity Index) - higher is better (range [0,1])
    ssim_values = []
    for i in range(len(pred_flat)):
        ssim_val = ssim(gt_flat[i], pred_flat[i], channel_axis=2, data_range=255)
        ssim_values.append(ssim_val)
    avg_ssim = np.mean(ssim_values)
    
    # 5. LPIPS (Learned Perceptual Image Patch Similarity) - lower is better
    # This requires a pre-trained network, so we'll use a simplified version
    # For full LPIPS, you'd need: import lpips; lpips_fn = lpips.LPIPS(net='alex')
    try:
        import lpips
        import torch
        lpips_fn = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')
        
        # Convert to torch tensors in range [-1, 1] with shape (N, 3, H, W)
        pred_torch = torch.from_numpy(pred_float / 127.5 - 1.0).permute(0, 3, 1, 2).float()
        gt_torch = torch.from_numpy(gt_float / 127.5 - 1.0).permute(0, 3, 1, 2).float()
        
        if torch.cuda.is_available():
            pred_torch = pred_torch.cuda()
            gt_torch = gt_torch.cuda()
        
        with torch.no_grad():
            lpips_values = lpips_fn(pred_torch, gt_torch)
        avg_lpips = lpips_values.mean().item()
    except:
        # Fallback if lpips not available
        avg_lpips = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips,
    }


def validate_video_generation(model, val_dataset, args, train_steps, videos_dir, id, accelerator, load_from_dataset=True):
    device = accelerator.device
    pipeline = model.module.pipeline if accelerator.num_processes > 1 else model.pipeline
    videos_row = args.video_num if not args.debug else 1
    videos_col = args.num_validation_batch

    # sample from val dataset
    batch_id = list(range(0,len(val_dataset),int(len(val_dataset)/videos_row/videos_col)))
    print(f"batch_id: {batch_id}")
    batch_id = batch_id[int(id*(videos_col)):int((id+1)*(videos_col))]
    batch_list = [val_dataset.__getitem__(id) for id in batch_id]
    video_gt = torch.cat([t['latent'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    text = [t['text'] for i,t in enumerate(batch_list)]
    actions = torch.cat([t['action'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    print(f"action shape: {actions.shape}")
    his_latent_gt, future_latent_ft = video_gt[:,:args.num_history], video_gt[:,args.num_history:]
    current_latent = future_latent_ft[:,0]
    print("image",current_latent.shape, 'action', actions.shape)
    if args.num_views == 1:
        assert  current_latent.shape[1:] == (4, 32, 32)
    elif args.num_views == 2:
        assert current_latent.shape[1:] == (4, 64, 32)
    else:
        assert current_latent.shape[1:] == (4, 96, 32)
    assert actions.shape[1:] == (int(args.num_frames+args.num_history), args.action_dim)

    # start generate
    with torch.no_grad():
        bsz = actions.shape[0]
        action_latent = model.module.action_encoder(actions, text, model.module.tokenizer, model.module.text_encoder, args.frame_level_cond) if accelerator.num_processes > 1 else model.action_encoder(actions, text, model.tokenizer, model.text_encoder,args.frame_level_cond) # (8, 1, 1024)
        print("action_latent",action_latent.shape)

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(args.num_views*args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type='latent',
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )
    
    pred_latents = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=args.num_views,n=1) # (B, 8, 4, 32,32)
    video_gt = torch.cat([his_latent_gt, future_latent_ft], dim=1) # (B, 8, 4, 32,32)
    video_gt = einops.rearrange(video_gt, 'b f c (m h) (n w) -> (b m n) f c h w', m=args.num_views, n=1) # (B, 8, 4, 32,32)
    
    # decode latent
    if video_gt.shape[2] != 3:  
        decoded_video = []
        bsz,frame_num = video_gt.shape[:2]
        video_gt = video_gt.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,video_gt.shape[0],args.decode_chunk_size):
            chunk = video_gt[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        video_gt = torch.cat(decoded_video,dim=0)
        video_gt = video_gt.reshape(bsz,frame_num,*video_gt.shape[1:])
        
        decoded_video = []
        bsz,frame_num = pred_latents.shape[:2]
        pred_latents = pred_latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,pred_latents.shape[0],args.decode_chunk_size):
            chunk = pred_latents[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])

    video_gt = ((video_gt / 2.0 + 0.5).clamp(0, 1)*255)
    video_gt = video_gt.to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)
    videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
    videos = videos.to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)
    
    # Compute multiple metrics between predicted and ground truth frames (before concatenation)
    # Only compare future frames (skip history frames)
    gt_frames_only = video_gt[:, args.num_history:]  # Corresponding ground truth frames
    
    metrics = compute_video_metrics(videos, gt_frames_only)
    
    # Log metrics to wandb
    accelerator.log({
        f"val_psnr_{id}": metrics['psnr'],
        f"val_ssim_{id}": metrics['ssim'],
        f"val_mse_{id}": metrics['mse'],
        f"val_mae_{id}": metrics['mae'],
        f"val_lpips_{id}": metrics['lpips'],
    }, step=train_steps)
    
    print(f"Video Metrics - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}, "
          f"MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}, LPIPS: {metrics['lpips']:.4f}")
    
    videos = np.concatenate([video_gt[:, :args.num_history],videos],axis=1) #(2,16,512,256,3)
    videos = np.concatenate([video_gt,videos],axis=-3) #(2,16,512,256,3)
    videos = np.concatenate([video for video in videos],axis=-2).astype(np.uint8) # (16,512,256*batch,3)

    # save and upload these videos to wandb
    os.makedirs(f"{videos_dir}/samples", exist_ok=True)
    filename = f"{videos_dir}/samples/train_steps_{train_steps}_{id}.mp4"
    mediapy.write_video(filename, videos, fps=2)
    
    # Upload video to wandb using accelerator
    accelerator.log({
        f"video_{args.tag}_{id}": wandb.Video(filename, fps=2, format="mp4")
    }, step=train_steps)
    
    return 


def main_val(args, validation_dataset=None):
    accelerator = Accelerator(
        log_with='wandb',
    )
    model = CrtlWorld(args)

    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        tag = args.tag
        run_name = f"val_{now}_{tag}"
        accelerator.init_trackers(args.wandb_project_name,config={}, init_kwargs={"wandb":{"name":run_name}})
        # count parameters num in each part
        num_params = sum(p.numel() for p in model.unet.parameters())
        print(f"Number of parameters in the unet: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.vae.parameters())
        print(f"Number of parameters in the vae: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"Number of parameters in the image_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.text_encoder.parameters())
        print(f"Number of parameters in the text_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.action_encoder.parameters())
        print(f"Number of parameters in the action_encoder: {num_params/1000000:.2f}M")

    # load form val_model_path
    print("load from val_model_path",args.val_model_path)
    model.load_state_dict(torch.load(args.val_model_path))
    model.to(accelerator.device)
    model.eval()
    validate_video_generation(model, validation_dataset, args, 0, 'output', 0, accelerator, load_from_dataset=False)

    # delete model to free up memory
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    checkpoint_paths = [
        "/data/Ctrl-World/model_ckpt/orca_dataset/checkpoint-20000.pt"
    ]

    # general settings
    args_train_case = wm_orca_args()
    args_train_case.num_views = 3
    args_train_case.max_num_samples = 1
    args_train_case.dataset_root_path = "dataset_example"
    args_train_case.dataset_names = "orca_dataset"
    args_train_case.dataset_meta_info_path = "dataset_meta_info"
    args_train_case.video_num = 1
    args_train_case.num_validation_batch = 1

    args_10_frames_case = copy.deepcopy(args_train_case) 
    args_10_frames_case.num_frames = 10
    args_10_frames_case.tag = "10_frames"

    args_20_frames_case = copy.deepcopy(args_train_case)
    args_20_frames_case.num_frames = 20
    args_20_frames_case.tag = "20_frames"

    args_30_frames_case = copy.deepcopy(args_train_case)
    args_30_frames_case.num_frames = 30
    args_30_frames_case.tag = "30_frames"

    args_40_frames_case = copy.deepcopy(args_train_case)
    args_40_frames_case.num_frames = 40
    args_40_frames_case.tag = "40_frames"

    args_history_0_case = copy.deepcopy(args_train_case)
    args_history_0_case.num_history = 0
    args_history_0_case.tag = "history_0"

    args_history_10_case = copy.deepcopy(args_train_case)
    args_history_10_case.num_history = 10
    args_history_10_case.tag = "history_10"

    args_history_20_case = copy.deepcopy(args_train_case)
    args_history_20_case.num_history = 20
    args_history_20_case.tag = "history_20"

    test_cases = [
        args_train_case,
        args_10_frames_case,
        args_20_frames_case,
        args_30_frames_case,
        args_40_frames_case,
        args_history_0_case,
        args_history_10_case,
        args_history_20_case,
    ]
    
    for checkpoint in checkpoint_paths:
        for args in test_cases:
            args.val_model_path = checkpoint
            print("Validating with settings: num_frames =", args.num_frames, ", num_history =", args.num_history)
            validation_dataset = Dataset_mix(args, mode='train')
            main_val(args, validation_dataset)