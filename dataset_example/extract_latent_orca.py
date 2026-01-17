import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mediapy
import argparse
from diffusers.models import AutoencoderKL
import torch
import numpy as np
import json
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset

import pandas as pd
from accelerate import Accelerator
from dotenv import load_dotenv
import requests
import datetime

load_dotenv()

def send_discord_message(message: str):
    """Sends a message to a Discord channel via a webhook."""
    # Using an environment variable for the webhook URL is a good practice
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Warning: DISCORD_WEBHOOK_URL environment variable not set. Skipping notification.")
        return

    data = {"content": message}
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to send Discord notification: {e}")


def parse_tuple(s):
    """Parse a string representation of a tuple into a tuple of integers."""
    try:
        # Remove parentheses and split by comma
        s = s.strip('()')
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("Tuple must be in format: (width,height) or width,height")


class EncodeLatentDataset(Dataset): 
    def __init__(self, args, old_path, new_path, vae, size=(192, 320), rgb_skip=3, val_ratio=0.3):
        self.args = args
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.vae = vae
        self.val_ratio = val_ratio

        annotation_files = [
            old_path + "/annotation/" + f for f in os.listdir(old_path + '/annotation') 
            if os.path.isfile(os.path.join(old_path + '/annotation', f))
        ]
        
        annotation_files.sort(
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        # # randomize order of annotation files
        np.random.shuffle(annotation_files)

        self.data = []
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                self.data.append(json.loads(f.read()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_data = self.data[idx]
        instruction = traj_data['texts'][0]
        traj_id = traj_data['episode_id']

        # Use hash for better distribution, then threshold based on val_ratio
        data_type = 'val' if (hash(traj_id) % 100) < (self.val_ratio * 100) else 'train'
        length = len(traj_data['observation.state.cartesian_position'])

        obs_car = []
        obs_joint =[]
        obs_hand_joint = []
        action_car = []
        action_hand_joint = []

        for i in range(length):
            obs_car.append(traj_data['observation.state.cartesian_pose'][i])
            obs_joint.append(traj_data['observation.state.joint_position'][i])
            obs_hand_joint.append(traj_data['observation.state.hand_joint_position'][i])
            action_car.append(traj_data['action.cartesian'][i])
            action_hand_joint.append(traj_data['action.hand_joint_position'][i])
        success = traj_data['success']
        video_paths = [
                    f'{self.old_path}/videos/{traj_id}/0_rgb.mp4',
                    f'{self.old_path}/videos/{traj_id}/1_rgb.mp4',
                    f'{self.old_path}/videos/{traj_id}/2_rgb.mp4']

        seg_video_paths = [
            f'{self.old_path}/videos/{traj_id}/0_segmentation.mp4',
            f'{self.old_path}/videos/{traj_id}/1_segmentation.mp4',
            f'{self.old_path}/videos/{traj_id}/2_segmentation.mp4',
        ]

        traj_info = {'success': success,
                     'observation.state.cartesian_position': obs_car,
                     'observation.state.joint_position': obs_joint,
                     'observation.state.hand_joint_position': obs_hand_joint,
                     'action.cartesian_position': action_car,
                     'action.hand_joint_position': action_hand_joint,
                    }
        

        # if f"{save_root}/videos/{data_type}/{traj_id}" exist, skip this trajectory
        try:
            self.process_traj(video_paths, seg_video_paths, traj_info, instruction, self.new_path, traj_id=traj_id, data_type=data_type, size=self.size, rgb_skip=self.skip, device=self.vae.device)
        except Exception as e:
            print(f"Error processing trajectory {traj_id}, skipping... Error: {e}")
            return 0
    
        return 0

    def downsample_binary_mask(self, video_id, seg_video_path, data_type, save_root, traj_id, rgb_skip, target_height, target_width, folder_name='downsampled_binary_masks'):
        """
        Downsample a binary mask to the target height and width using nearest neighbor interpolation.
        
        Args:
            mask: Input binary mask as a numpy array or tensor of shape (T, H, W, 1)
            target_height: Desired height after downsampling
            target_width: Desired width after downsampling
            weight: Weight to multiply the downsampled mask
            
        Returns:
            Downsampled mask tensor of shape (B, 1, target_height, target_width)
        """
        seg_video = mediapy.read_video(seg_video_path)

        mask = torch.tensor(seg_video).permute(0, 3, 1, 2)
        mask = torch.sum(mask, dim=-3)  # Keep only one channel for binary mask
        mask = mask.unsqueeze(-1)  # Add channel dimension back

        # reduce temporal length
        mask = mask[::rgb_skip]

        # Convert to tensor if it's a numpy array
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()
        
        # Ensure mask is in the correct range [0, 1]
        if torch.max(mask) > 1.0:
            mask = mask / torch.max(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0)
        
        # Convert from (T, H, W, 1) to (T, 1, H, W) for interpolate function
        # PyTorch's interpolate expects (N, C, H, W) format
        mask = mask.permute(0, 3, 1, 2)  # (T, H, W, 1) -> (T, 1, H, W)
        
        # Downsample using nearest neighbor interpolation
        downsampled_mask = torch.nn.functional.interpolate(
            mask, 
            size=(target_height, target_width), 
            mode='nearest'
        )

        os.makedirs(f"{save_root}/{folder_name}/{data_type}/{traj_id}", exist_ok=True)
        torch.save(downsampled_mask, f"{save_root}/{folder_name}/{data_type}/{traj_id}/{video_id}.pt")

        return len(mask)
        
    def extract_rgb_latents(self, video_id, video_path, save_root, traj_id, data_type, size, rgb_skip, device, folder_name='latent_videos'):
        # load and resize video and save
        video = mediapy.read_video(video_path)

        if "segmentation" in folder_name:
            # save the video to the new folder
            os.makedirs(f"{save_root}/segmentation_videos/{data_type}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{save_root}/segmentation_videos/{data_type}/{traj_id}/{video_id}.mp4", video[::rgb_skip], fps=self.args.original_fps / self.args.down_sample)
            print("segmentation video")

        frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
        print("frames shape: ", frames.shape)
        frames = frames[::rgb_skip]  # Skip frames to save memory here!!!
        x = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)
        resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
        resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        os.makedirs(f"{save_root}/videos/{data_type}/{traj_id}", exist_ok=True)
        mediapy.write_video(f"{save_root}/videos/{data_type}/{traj_id}/{video_id}.mp4", resize_video, fps=self.args.original_fps / rgb_skip)

        # save svd latent
        x = x.to(device)
        with torch.no_grad():
            batch_size = 64
            latents = []
            # Handle both wrapped and unwrapped VAE models
            vae_model = self.vae.module if hasattr(self.vae, 'module') else self.vae
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                latent = vae_model.encode(batch).latent_dist.sample().mul_(vae_model.config.scaling_factor).cpu()
                latents.append(latent)
            x = torch.cat(latents, dim=0)
        os.makedirs(f"{save_root}/{folder_name}/{data_type}/{traj_id}", exist_ok=True)
        torch.save(x, f"{save_root}/{folder_name}/{data_type}/{traj_id}/{video_id}.pt")
        return len(frames)
    
    def process_traj(self, video_paths, seg_video_paths, traj_info, instruction, save_root,traj_id=0,data_type='val', size=(192,320), rgb_skip=3, device='cuda'):
        for video_id, (video_path, seg_video_path) in enumerate(zip(video_paths, seg_video_paths)):
            # extract rgb latents using VAE
            video_length = self.extract_rgb_latents(video_id, video_path, save_root, traj_id, data_type, size, rgb_skip, device, folder_name='latent_videos')

            if self.args.encode_segmentation_with_svd:
                self.extract_rgb_latents(video_id, seg_video_path, save_root, traj_id, data_type, size, rgb_skip, device, folder_name='latent_segmentation_videos')

            # get downsample binary mask using interpolate
            if self.args.use_hand_mask:
                seg_length = self.downsample_binary_mask(video_id, seg_video_path, data_type, save_root, traj_id, rgb_skip, int(self.args.height / self.args.vae_compression_rate), int(self.args.width / self.args.vae_compression_rate))
                assert seg_length == video_length
        
        # record cartesain aligned with video frames
        cartesian_pose = np.array(traj_info['observation.state.cartesian_position'])
        cartesian_hand_joint = np.array(traj_info['observation.state.hand_joint_position'])
        cartesian_states = np.concatenate((cartesian_pose, cartesian_hand_joint),axis=-1)[::rgb_skip].tolist()
        
        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info['success']),
            "video_length": video_length,
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info['observation.state.cartesian_position']),
            "videos": [
                {"video_path": f"videos/{data_type}/{traj_id}/0.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/1.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/2.mp4"}
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/2.pt"}
            ],
            'states': cartesian_states,
            'observation.state.cartesian_position': traj_info['observation.state.cartesian_position'],
            'observation.state.joint_position': traj_info['observation.state.joint_position'],
            'observation.state.hand_joint_position': traj_info['observation.state.hand_joint_position'],
            'action.cartesian_position': traj_info['action.cartesian_position'],
            'action.hand_joint_position': traj_info['action.hand_joint_position'],
        }

        if seg_video_paths is not None:
            info['latent_segmentation_videos'] = [
                {"latent_video_path": f"latent_segmentation_videos/{data_type}/{traj_id}/0.pt"},
                {"latent_video_path": f"latent_segmentation_videos/{data_type}/{traj_id}/1.pt"},
                {"latent_video_path": f"latent_segmentation_videos/{data_type}/{traj_id}/2.pt"},
            ]
            info['segmentation_videos'] = [
                {"video_path": f"segmentation_videos/{data_type}/{traj_id}/0.mp4"},
                {"video_path": f"segmentation_videos/{data_type}/{traj_id}/1.mp4"},
                {"video_path": f"segmentation_videos/{data_type}/{traj_id}/2.mp4"},
            ]

        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{traj_id}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    from config import wm_orca_args
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--orca_dataset_path', type=str, default='/data/Ctrl-World/datasets')
    parser.add_argument('--orca_output_path', type=str, default='/data/Ctrl-World/datasets')
    parser.add_argument('--svd_path', type=str, default='stabilityai/stable-video-diffusion-img2vid')
    parser.add_argument('--frame_size', type=parse_tuple, default=(256, 256))
    parser.add_argument('--use_hand_mask', action='store_true')
    parser.add_argument('--encode_segmentation_with_svd', type=bool, default=True)
    parser.add_argument('--val_ratio', type=float, default=0.3)
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    wm_arguments = wm_orca_args()

    if args.use_hand_mask:
        wm_arguments.use_hand_mask = True

    if args.encode_segmentation_with_svd:
        wm_arguments.encode_segmentation_with_svd = True

    data_list = [
        {
            'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2026-01-03T18-18-58/data_working_hand_mask_sine_hand_motions',
            'desired_fps': 10,
            'folder_name': 'data_working_hand_mask_sine_hand_motions'
        },
        {
            'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2026-01-03T18-18-58/data_working_hand_mask_sine_hand_motions_fixed_ee',
            'desired_fps': 10,
            'folder_name': 'data_working_hand_mask_sine_hand_motions_fixed_ee'
        }
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_close_mimicgen_hand_mask',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_close_mimicgen_hand_mask',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_close_sine_hand_mask',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_close_sine_hand_mask',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_object_collisions_open_close',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_fixed_ee_object_collisions_open_close',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_open_close',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_fixed_ee_open_close',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_random',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_fixed_ee_random',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_sine_object_collision_high_res',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_fixed_ee_sine_object_collision_high_res',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_sine_object_collision_high_res',
        #     'desired_fps': 25,
        #     'folder_name': 'data_fix_cam_fixed_ee_sine_object_collision_high_res',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_sine_object_collision_high_res',
        #     'desired_fps': 50,
        #     'folder_name': 'data_fix_cam_fixed_ee_sine_object_collision_high_res',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_sine_object_collisions',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_fixed_ee_sine_object_collisions',
        # },
        # { 
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_sine_object_collisions',
        #     'desired_fps': 25,
        #     'folder_name': 'data_fix_cam_fixed_ee_sine_object_collisions',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_ee_sine_object_collisions',
        #     'desired_fps': 50,
        #     'folder_name': 'data_fix_cam_fixed_ee_sine_object_collisions',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_fixed_fingers_random_object_locations',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_fixed_fingers_random_object_locations',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_fix_cam_view_close_random_motions',
        #     'desired_fps': 10,
        #     'folder_name': 'data_fix_cam_view_close_random_motions',
        # },
        # {
        #     'data_path': '/data/faive_lab/datasets/converted_to_lerobot/2025-12-23T16-08-41/data_sine_fixed_ee',
        #     'desired_fps': 10,
        #     'folder_name': 'data_sine_fixed_ee',
        # },
    ]

    accelerator = Accelerator()
    
    # Load VAE once and prepare with accelerator
    if accelerator.is_main_process:
        print(f"Loading VAE from {args.svd_path}...")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Current process index: {accelerator.process_index}")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae")
    vae = accelerator.prepare(vae)
    vae.eval()
    
    # Process each dataset in data_dict
    now = str(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    for dataset_config in data_list:
        dataset_path = dataset_config['data_path']
        desired_fps = dataset_config['desired_fps']
        folder_name = dataset_config['folder_name']
        
        # Calculate rgb_skip based on desired FPS
        # Assuming original_fps is 50 (adjust if different)
        original_fps = wm_arguments.original_fps
        rgb_skip = int(original_fps / desired_fps)
        
        # Create output path for this specific dataset
        output_path = os.path.join(os.path.join(args.orca_output_path, now), folder_name + f'_{desired_fps}fps')
        os.makedirs(output_path, exist_ok=True)
        
        start_time = datetime.datetime.now()
        if accelerator.is_main_process:
            start_msg = f"🚀 **Starting Dataset Processing**\n" \
                       f"📁 Dataset: `{folder_name}`\n" \
                       f"🎯 Desired FPS: {desired_fps} Hz\n" \
                       f"⏰ Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            send_discord_message(start_msg)
            
            print(f"\n{'='*80}")
            print(f"Processing dataset: {folder_name}")
            print(f"Source path: {dataset_path}")
            print(f"Output path: {output_path}")
            print(f"Original FPS: {original_fps} Hz")
            print(f"Desired FPS: {desired_fps} Hz")
            print(f"RGB skip: {rgb_skip}")
            print(f"Actual FPS after skipping: {original_fps / rgb_skip} Hz")
            print(f"{'='*80}\n")
        
        dataset = EncodeLatentDataset(
            args=wm_arguments,
            old_path=dataset_path,
            new_path=output_path,
            vae=vae,
            size=args.frame_size,
            rgb_skip=rgb_skip,
            val_ratio=args.val_ratio,
        )
        
        tmp_data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=0,  # Keep 0 because VAE is used in __getitem__
                pin_memory=True,
            )
        # Accelerate will automatically distribute dataset across GPUs
        tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)

        for idx, _ in enumerate(tmp_data_loader):
            if idx == 1 and args.debug:
                break
            if idx % 100 == 0 and accelerator.is_main_process:
                print(f"[{folder_name}] Precomputed {idx} samples", flush=True)
        
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        if accelerator.is_main_process:
            end_msg = f"✅ **Completed Dataset Processing**\n" \
                     f"📁 Dataset: `{folder_name}`\n" \
                     f"🎯 Desired FPS: {desired_fps} Hz\n" \
                     f"⏰ Finish Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n" \
                     f"⏱️ Duration: {str(duration).split('.')[0]}"
            send_discord_message(end_msg)
            print(f"\nCompleted processing {folder_name}\n")

# accelerate launch dataset_example/extract_latent.py --droid_hf_path /cephfs/shared/droid_hf/droid_1.0.1 --droid_output_path dataset_example/droid_subset --svd_path /cephfs/shared/llm/stable-video-diffusion-img2vid --debug

