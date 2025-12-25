# from diffusers import StableVideoDiffusionPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.ctrl_world import CrtlWorld

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
import mediapy
from models.ctrl_world import CrtlWorld
from config import wm_orca_args
import math


def main(args):
    logger = get_logger(__name__, log_level="INFO")
    
    # allows you to log when using multiple GPUs
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_dir=args.output_dir
    )

    # Loading model checkpoint if one is given otherwise initialize a new model
    model = CrtlWorld(args)
    if args.ckpt_path is not None:
        print(f"Loading checkpoint from {args.ckpt_path}!")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    model.to(accelerator.device)
    model.train()

    # using AdamW as optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Showing a few intersting stats about the model that we're about to train
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        tag = args.tag
        run_name = f"train_{now}_{tag}"
        # Convert args to dict for wandb
        config_dict = args_to_dict(args)
        accelerator.init_trackers(args.wandb_project_name, config=config_dict, init_kwargs={"wandb":{"name":run_name}})
        os.makedirs(args.output_dir, exist_ok=True)
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

    # Loading both the train and validation dataset
    from dataset.dataset_orca import Dataset_mix
    train_dataset = Dataset_mix(args,mode='train')
    print(f"Number of training samples: {len(train_dataset)}")
    val_dataset = Dataset_mix(args,mode='val')
    print(f"Number of validation samples: {len(val_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size,
        shuffle=args.shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.train_batch_size,
        shuffle=args.shuffle
    )

    # Prepare everything with our accelerator
    # does the GPU distribution
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
   
    ############################ training ##############################
    # printing training information
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_train_epochs = math.ceil(args.max_train_steps * args.gradient_accumulation_steps*total_batch_size / len(train_dataloader))
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")


    # Initialize training counters
    global_step = 0  # Counts optimizer steps (after gradient accumulation)
    forward_step = 0  # Counts forward passes (can be multiple per optimizer step)
    train_loss = 0.0  # Accumulates loss for logging
    # Create progress bar (only on main process to avoid duplicates)
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Main training loop over epochs
    for epoch in range(num_train_epochs):
        # Iterate through batches in the training dataloader
        for step, batch in enumerate(train_dataloader):
            # Context manager handles gradient accumulation logic
            with accelerator.accumulate(model):
                # Enable automatic mixed precision (fp16/bf16) for faster training
                with accelerator.autocast():
                    # Forward pass: compute loss
                    loss_gen, _ = model(batch)
                # Gather losses from all GPUs and compute mean for logging
                avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                # Accumulate loss (divide by grad accum steps for correct averaging)
                train_loss += avg_loss.item()/ args.gradient_accumulation_steps
                # Backward pass: compute gradients (handles multi-GPU synchronization)
                accelerator.backward(loss_gen)
                # Get model parameters for gradient clipping
                params_to_clip = model.parameters()
                # Check if gradients should be synchronized (true after gradient_accumulation_steps)
                if accelerator.sync_gradients:
                    # Clip gradients to prevent exploding gradients
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                # Update model weights
                optimizer.step()
                # Clear gradients for next iteration
                optimizer.zero_grad()
                # Increment forward pass counter
                forward_step += 1

            # LOGGING 
            # Only execute when gradients have been synchronized (i.e., actual optimizer step)
            if accelerator.sync_gradients:
                # Update progress bar
                progress_bar.update(1)
                # Increment global step counter
                global_step += 1
                # Log average loss every 100 steps
                if global_step %100 == 0:
                    # Update progress bar display with current loss
                    progress_bar.set_postfix({"loss": train_loss})
                    # Log to wandb
                    accelerator.log({"train_loss": train_loss/100}, step=global_step)
                    # Reset accumulated loss
                    train_loss = 0.0
                # Save model checkpoint at specified intervals (only on main process)
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    # Create checkpoint save path
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    # Unwrap model from DDP wrapper and save state dict
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                # Generate validation videos at specified intervals (only on main process)
                if global_step % args.validation_steps == 5 and accelerator.is_main_process:
                    # Switch to evaluation mode (disables dropout, etc.)
                    model.eval()
                    # Use autocast for inference
                    with accelerator.autocast():
                        # Generate multiple validation videos
                        for id in range(args.video_num):
                            validate_video_generation(model, val_dataset, args,global_step, args.output_dir, id, accelerator)
                    # Switch back to training mode
                    model.train()



def main_val(args):
    accelerator = Accelerator()
    model = CrtlWorld(args)
    # load form val_model_path
    print("load from val_model_path",args.val_model_path)
    model.load_state_dict(torch.load(args.val_model_path))
    model.to(accelerator.device)
    model.eval()
    validate_video_generation(model, None, args, 0, 'output', 0, accelerator, load_from_dataset=False)
    
            

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
    
    # Compute metrics over a 1-second time window for fair comparison across different fps
    metrics = compute_video_metrics(videos, gt_frames_only, fps=args.fps, time_window_seconds=1.0)
    
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
        f"video_{id}": wandb.Video(filename, fps=2, format="mp4")
    }, step=train_steps)
    
    return 


def compute_video_metrics(pred_frames, gt_frames, fps=10, time_window_seconds=1.0):
    """
    Compute multiple metrics to evaluate predicted video quality against ground truth
    over a specific time window.
    
    Args:
        pred_frames: (B, T, H, W, 3) predicted frames in range [0, 255], uint8
        gt_frames: (B, T, H, W, 3) ground truth frames in range [0, 255], uint8
        fps: frame rate of the video (default: 10)
        time_window_seconds: time window in seconds to evaluate (default: 1.0)
    
    Returns:
        dict with metrics: psnr, ssim, mse, mae, lpips
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Calculate number of frames corresponding to the time window
    frames_in_window = int(fps * time_window_seconds)
    total_frames = pred_frames.shape[1]
    
    # Ensure we don't exceed available frames
    frames_to_evaluate = min(frames_in_window, total_frames)
    
    # Slice to only evaluate the specified time window (first N frames)
    pred_frames = pred_frames[:, :frames_to_evaluate]
    gt_frames = gt_frames[:, :frames_to_evaluate]
    
    print(f"Evaluating metrics over {frames_to_evaluate} frames ({frames_to_evaluate/fps:.2f}s at {fps} fps)")
    
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


def args_to_dict(args):
    """Convert args object to dictionary for wandb config."""
    config_dict = {}
    for attr_name in dir(args):
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
                value = dtype_str_map.get(value, str(value))
            config_dict[attr_name] = value
        except Exception:
            continue
    return config_dict

if __name__ == "__main__":
    # reset parameters with command line
    from argparse import ArgumentParser
    from utils.config_loader import load_experiment_config, save_config_to_yaml, list_available_experiments

    parser = ArgumentParser()

    # Main way to specify experiment config
    parser.add_argument('--config', type=str, default=None,
                       help='Direct path to experiment config file')
    parser.add_argument('--list_experiments', action='store_true',
                       help='List all available experiment configurations')    

    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    
    args_cli = parser.parse_args()
    # List experiments if requested
    if args_cli.list_experiments:
        print("\n" + "="*60)
        print("Available Experiments:")
        print("="*60)
        experiments = list_available_experiments()
        for exp in experiments:
            print(f"\n📋 {exp['name']}")
            print(f"   File: {exp['file']}")
            print(f"   Description: {exp['description']}")
        print("\n" + "="*60)
        exit(0)

    args = wm_orca_args()

    # Load experiment config if specified
    if args_cli.config:
        print(f"Loading experiment config from: {args_cli.config}")
        args = load_experiment_config(args_cli.config, args)
    else:
        print("No experiment config specified, using default config")
        exit(0)
    
    # Apply command-line overrides (these take precedence)
    for key, value in vars(args_cli).items():
        if value is not None and hasattr(args, key):
            print(f"Overriding {key}: {getattr(args, key)} -> {value}")
            setattr(args, key, value)
    
    # Compute derived values
    if args_cli.fps is not None:
        args.down_sample = int(args.original_fps / args_cli.fps)
    
    if args_cli.tag is not None:
        args.output_dir = f"model_ckpt/{args_cli.tag}"
        args.wandb_run_name = args_cli.tag
    
    # Save the final config used for this run
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    main(args)

    # CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline accelerate launch --main_process_port 29501 train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info
    # CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29506 unit_test2.py

    # args = Args()
    # from video_dataset.dataset_droid_exp33 import Dataset_mix
    # dataset = Dataset_mix(args,mode='val')
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)
    # model = CrtlWorld(args).to('cuda')
    # # print model parameter num
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    # total_elements = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    # print(f"Total number of learnable parameters: {total_elements}")
    # model.train()
    

    # for batch in dataloader:
    #     print(batch['latent'].shape)
    #     print(batch['text'])
    #     print(batch['action'].shape)

    #     loss,_ = model(batch)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print(loss.item())





    # device = 'cuda'
    # video_encoder = VideoEncoder(hidden_size=1024).to(device)
    # # count the parameters of the model
    # num_params = sum(p.numel() for p in video_encoder.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # vae_latent = torch.randn(8, 1, 4, 32, 32).to(device)
    # clip_latent = torch.randn(8, 20, 512).to(device)
    # current_img = video_encoder(vae_latent, clip_latent)
    # print(current_img.shape)  # (8, 1, 4, 32, 32)


    # pos_emb = get_2d_sincos_pos_embed(1024, 16)
    # print(pos_emb.shape)  # (256, 1024)
    # clip_emb = get_1d_sincos_pos_embed_from_grid(1024, np.arange(20))
    # print(clip_emb.shape)  # (20, 512)
