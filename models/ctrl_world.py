# from diffusers import StableVideoDiffusionPipeline
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

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


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Action_encoder2(nn.Module):
    def __init__(self, action_dim, action_num, hidden_sizes, text_cond=True):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self.text_cond = text_cond

        input_dim = int(action_dim)

        # Create hidden layers: flatten list comprehension properly
        hidden_layers = []
        for hidden_size in self.hidden_sizes:
            hidden_layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
        
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, self.hidden_sizes[0]),
            nn.SiLU(),
            *hidden_layers,
            nn.Linear(self.hidden_sizes[-1], 1024),
        )
        # kaiming initialization
        nn.init.kaiming_normal_(self.action_encode[0].weight, mode='fan_in', nonlinearity='relu')
        if len(hidden_layers) > 0:
            nn.init.kaiming_normal_(self.action_encode[2].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, action,  texts=None, text_tokinizer=None, text_encoder=None, frame_level_cond=True,):
        # action: (B, action_num, action_dim)
        B,T,D = action.shape
        if not frame_level_cond:
            action = einops.rearrange(action, 'b t d -> b 1 (t d)')
        action = self.action_encode(action)

        if texts is not None and self.text_cond:
            # with 50% probability, add text condition
            with torch.no_grad():
                inputs = text_tokinizer(texts, padding='max_length', return_tensors="pt", truncation=True).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                hidden_text = outputs.text_embeds # (B, 512)
                hidden_text = einops.repeat(hidden_text, 'b c -> b 1 (n c)', n=2) # (B, 1, 1024)
            
            action = action + hidden_text # (B, T, hidden_size)
        return action # (B, 1, hidden_size) or (B, T, hidden_size) if frame_level_cond


class DinoV2VisualActionEncoder(nn.Module):
    # it takes a sequnce of RGB frames which represent the segmentation of the hand. (B, T, C, H, W)
    def __init__(self, num_in_channels=3, embed_dim=1024, hub_dir="/data/hub", dinov2_size=224):
        super().__init__()
        
        # Set custom directory for torch.hub cache if provided
        if hub_dir is not None:
            torch.hub.set_dir(hub_dir)
        
        # Pretrained spatial encoder (frozen)
        self.spatial_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for param in self.spatial_backbone.parameters():
            param.requires_grad = False
        
        # DINOv2 requires input size divisible by patch size (14)
        # Common sizes: 224 (224/14=16), 280 (280/14=20), 392 (392/14=28)
        assert dinov2_size % 14 == 0, f"dinov2_size must be divisible by 14 (patch size), got {dinov2_size}"
        self.dinov2_size = dinov2_size
        
        # Trainable task-specific adapter with learnable downsampling
        # Input: (B, num_in_channels, H, W) -> Output: (B, 3, dinov2_size, dinov2_size)
        # Strategy: Use learnable convolutions to downsample to target size
        # The adapter learns features at multiple scales, then adjusts to target size
        
        # Calculate kernel size for final size adjustment (for common case of 256->224)
        # Formula: output_size = (input_size - kernel_size + 2*padding) / stride + 1
        # For 256->224 with stride=1, padding=0: kernel_size = 256 - 224 + 1 = 33
        # We'll use this for the expected input size, but handle variable sizes in forward
        expected_input_size = 256  # Common case
        if expected_input_size > dinov2_size:
            final_kernel_size = expected_input_size - dinov2_size + 1
        else:
            final_kernel_size = 1
        
        self.seg_adapter = nn.Sequential(
            # First conv: channel transformation
            nn.Conv2d(num_in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Learnable downsampling: use strided conv to reduce spatial size
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # H/2, W/2 (e.g., 256 -> 128)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Learnable upsampling: transposed conv to get back to larger size
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Final channel reduction to num_in_channels
            nn.Conv2d(64, num_in_channels, 1),
        )
        
        # Learnable size adjustment layer (applied conditionally in forward)
        # For 256->224: kernel_size=33, stride=1, padding=0 gives (256-33+1)=224
        if final_kernel_size > 1:
            self.size_adjust_conv = nn.Conv2d(num_in_channels, num_in_channels, kernel_size=final_kernel_size, stride=1, padding=0)
        else:
            self.size_adjust_conv = None
        
        # Trainable temporal reasoning module
        self.temporal_module = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048),
            num_layers=2
        )
        
        # Trainable projection
        self.proj = nn.Linear(768, embed_dim)
    
    def forward(self, seg_sequence):
        B, T, C, H, W = seg_sequence.shape
        
        # Extract spatial features with frozen DINOv2
        spatial_feats = []
        for t in range(T):
            x = self.seg_adapter(seg_sequence[:, t])  # (B, 3, H, W) - after adapter, size is back to ~HxW
            
            # Apply learnable size adjustment if available (for exact size matching)
            if self.size_adjust_conv is not None and x.shape[2] == H and x.shape[3] == W:
                # Use learnable conv to adjust size (e.g., 256 -> 224)
                x = self.size_adjust_conv(x)  # (B, 3, dinov2_size, dinov2_size)
            elif x.shape[2] != self.dinov2_size or x.shape[3] != self.dinov2_size:
                # Fallback to interpolation if size doesn't match (for variable input sizes)
                x = torch.nn.functional.interpolate(
                    x, 
                    size=(self.dinov2_size, self.dinov2_size), 
                    mode='bilinear', 
                    align_corners=False
                )  # (B, 3, dinov2_size, dinov2_size)
            
            feat = self.spatial_backbone(x)
            spatial_feats.append(feat)
        
        spatial_feats = torch.stack(spatial_feats, dim=1)  # [B, T, 768]
        
        # Learn temporal patterns
        spatial_feats = spatial_feats.transpose(0, 1)  # [T, B, 768]
        temporal_feats = self.temporal_module(spatial_feats)
        temporal_feats = temporal_feats.transpose(0, 1)  # [B, T, 768]
        
        # Project to output
        output = self.proj(temporal_feats)
        
        return output

class CrtlWorld(nn.Module):
    def __init__(self, args):
        super(CrtlWorld, self).__init__()

        self.args = args

        # load from pretrained stable video diffusion
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(args.svd_model_path)
        # repalce the unet to support frame_level pose condition
        print("replace the unet to support action condition and frame_level pose!")
        unet = UNetSpatioTemporalConditionModel()
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        self.pipeline.unet = unet
        
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler

        # freeze vae, image_encoder, enable unet gradient ckpt
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.unet.enable_gradient_checkpointing()

        # SVD is a img2video model, load a clip text encoder
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.clip_model_path,use_fast=False)
        self.text_encoder.requires_grad_(False)

        # initialize an action projector
        if args.action_encoder == "dino_visual":
            self.action_encoder = DinoV2VisualActionEncoder(num_in_channels=3, embed_dim=1024)
        else:
            self.action_encoder = Action_encoder2(action_dim=args.action_dim, action_num=int(args.num_history+args.num_frames), hidden_sizes=args.action_encoder_hidden_dims, text_cond=args.text_cond)


    def forward(self, batch):
        latents = batch['latent'] # (B, 16, 4, 32, 32)
        latent_hand_masks = batch['hand_mask'] if self.args.use_hand_mask else None
        texts = batch['text']
        dtype = self.unet.dtype
        device = self.unet.device
        P_mean=0.7
        P_std=1.6
        noise_aug_strength = 0.0

        num_history  = self.args.num_history
        latents = latents.to(device) #[B, num_history + num_frames]

        # current img as condition image to stack at channel wise, add random noise to current image, noise strength 0.0~0.2
        current_img = latents[:,num_history:(num_history+1)] # (B, 1, 4, 32, 32)
        bsz,num_frames = latents.shape[:2]
        current_img = current_img[:,0] # (B, 4, 32, 32)
        sigma = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
        c_in = 1 / (sigma**2 + 1) ** 0.5
        current_img = c_in*(current_img + torch.randn_like(current_img) * sigma)
        condition_latent = einops.repeat(current_img, 'b c h w -> b f c h w', f=num_frames) # (8, 16,12, 32,32)
        if self.args.his_cond_zero:
            condition_latent[:, :num_history] = 0.0 # (B, num_history+num_frames, 4, 32, 32)


        # action condition
        action = batch['action'] # (B, f, 7)
        action = action.to(device)
        if self.args.action_encoder == "dino_visual":
            visual_actions = batch['visual_actions'] # (B, T, 3, 256, 256)
            visual_actions = visual_actions.to(device)
            action_hidden = self.action_encoder(visual_actions) # (B, T, 1024)
            action_hidden = action_hidden.to(device)
        else:
            action_hidden = self.action_encoder(action, texts, self.tokenizer, self.text_encoder, frame_level_cond=self.args.frame_level_cond) # (B, f, 1024)
            action_hidden = action_hidden.to(device)

        # for classifier-free guidance, with 5% probability, set action_hidden to 0
        uncond_hidden_states = torch.zeros_like(action_hidden)
        text_mask = (torch.rand(action_hidden.shape[0], device=device)>0.05).unsqueeze(1).unsqueeze(2)
        action_hidden = action_hidden*text_mask+uncond_hidden_states*(~text_mask)

        # diffusion forward process on future latent
        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out =  -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz])
        loss_weight = (sigma ** 2 + 1) / sigma ** 2
        noisy_latents = (latents + torch.randn_like(latents) * sigma)

        # add 0~0.3 noise to history, history as condition
        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
        history = latents[:,:num_history] # (B, num_history, 4, 32, 32)
        noisy_history = 1/(sigma_h**2+1)**0.5 *(history + sigma_h * torch.randn_like(history)) # (B, num_history, 4, 32, 32)
        input_latents = torch.cat([noisy_history, c_in*noisy_latents[:,num_history:]], dim=1) # (B, num_history+num_frames, 4, 32, 32)

        # svd stack a img at channel wise
        input_latents = torch.cat([input_latents, condition_latent/self.vae.config.scaling_factor], dim=2)
        motion_bucket_id = self.args.motion_bucket_id
        fps = self.args.fps
        added_time_ids = self.pipeline._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, action_hidden.dtype, bsz, 1, False)
        added_time_ids = added_time_ids.to(device)

        # forward unet
        loss = 0
        model_pred = self.unet(input_latents, c_noise, encoder_hidden_states=action_hidden, added_time_ids=added_time_ids,frame_level_cond=self.args.frame_level_cond).sample
        predict_x0 = c_out * model_pred + c_skip * noisy_latents 

        # only calculate loss on future frames
        pred_difference = predict_x0[:,num_history:] - latents[:,num_history:]
        if self.args.use_hand_mask:
            print("pred_difference shape: ", pred_difference.shape)
            print("latent_hand_masks: ", latent_hand_masks[:, num_history:].shape)

            pred_difference = pred_difference * latent_hand_masks[:, num_history:] * self.args.hand_weight + pred_difference * (1 - latent_hand_masks[:, num_history:])
        loss += (pred_difference**2 * loss_weight).mean()

        return loss, torch.tensor(0.0, device=device,dtype=dtype)
