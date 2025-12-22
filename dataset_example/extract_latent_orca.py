import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mediapy
import os
import argparse
from diffusers.models import AutoencoderKL
import mediapy
import torch
import numpy as np
import json
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
import mediapy
from torch.utils.data import Dataset

import pandas as pd
from accelerate import Accelerator


def parse_tuple(s):
    """Parse a string representation of a tuple into a tuple of integers."""
    try:
        # Remove parentheses and split by comma
        s = s.strip('()')
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("Tuple must be in format: (width,height) or width,height")


class EncodeLatentDataset(Dataset): 
    def __init__(self, args, old_path, new_path, svd_path, device, size=(192, 320), rgb_skip=3):
        self.args = args
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)

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

        data_type = 'val' if traj_id%100 == 3 else 'train'
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
        ] if self.args.use_hand_mask else None

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

    def downsample_binary_mask(self, video_id, seg_video_path, data_type, save_root, traj_id, rgb_skip, target_height, target_width):
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

        print(f"Max value in binary mask before processing: {torch.max(mask)}")

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
        print("Mask shape: ", mask.shape)
        
        # Downsample using nearest neighbor interpolation
        downsampled_mask = torch.nn.functional.interpolate(
            mask, 
            size=(target_height, target_width), 
            mode='nearest'
        )

        os.makedirs(f"{save_root}/latent_segmentation_videos/{data_type}/{traj_id}", exist_ok=True)
        torch.save(downsampled_mask, f"{save_root}/latent_segmentation_videos/{data_type}/{traj_id}/{video_id}.pt")

        print(f"Max value in binary mask after processing: {torch.max(downsampled_mask)}")

        # store a plot of the dwonsample mask
        import matplotlib.pyplot as plt
        print(downsampled_mask[-1,0].squeeze().shape)
        plt.imsave("test.png", downsampled_mask[0,0].squeeze().cpu().numpy(), cmap='gray')

        return len(mask)
        
    def extract_rgb_latents(self, video_id, video_path, save_root, traj_id, data_type, size, rgb_skip, device):
        # load and resize video and save
        video = mediapy.read_video(video_path)
        frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
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
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                # x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                latents.append(latent)
            x = torch.cat(latents, dim=0)
        os.makedirs(f"{save_root}/latent_videos/{data_type}/{traj_id}", exist_ok=True)
        torch.save(x, f"{save_root}/latent_videos/{data_type}/{traj_id}/{video_id}.pt")
        return len(frames)

    def process_traj(self, video_paths, seg_video_paths, traj_info, instruction, save_root,traj_id=0,data_type='val', size=(192,320), rgb_skip=3, device='cuda'):
        if seg_video_paths is not None:
            for video_id, (video_path, seg_video_path) in enumerate(zip(video_paths, seg_video_paths)):
                # extract rgb latents using VAE
                video_length = self.extract_rgb_latents(video_id, video_path, save_root, traj_id, data_type, size, rgb_skip, device)

                # get downsample binary mask using interpolate
                seg_length = self.downsample_binary_mask(video_id, seg_video_path, data_type, save_root, traj_id, rgb_skip, int(self.args.height / self.args.vae_compression_rate), int(self.args.width / self.args.vae_compression_rate))
                assert seg_length == video_length
        else:
            for video_id, video_path in enumerate(video_paths):
                self.extract_rgb_latents(video_id, video_path, save_root, traj_id, data_type, size, rgb_skip, device)
        
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

        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{traj_id}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    from config import wm_orca_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--orca_dataset_path', type=str, default='/data/Ctrl-World/dataset_example/lerobot_dataset')
    parser.add_argument('--orca_output_path', type=str, default='/data/Ctrl-World/dataset_example/orca_dataset')
    parser.add_argument('--svd_path', type=str, default='stabilityai/stable-video-diffusion-img2vid')
    parser.add_argument('--frame_size', type=parse_tuple, default=(256, 256))
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    wm_arguments = wm_orca_args()

    accelerator = Accelerator()
    print(f"Actual FPS after skipping: {wm_arguments.original_fps / wm_arguments.down_sample} Hz")
    dataset = EncodeLatentDataset(
        args=wm_arguments,
        old_path=args.orca_dataset_path,
        new_path= args.orca_output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=args.frame_size,
        rgb_skip=wm_arguments.down_sample, # NOTE: Synthetic data is recorded at 45 fps but the decimation is set to 2 which means the fps is actually 45, we use 5 fps
    )
    tmp_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
        )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    for idx, _ in enumerate(tmp_data_loader):
        if idx == 1 and args.debug:
            break
        if idx % 100 == 0 and accelerator.is_main_process:
            print(f"Precomputed {idx} samples", flush=True)

# accelerate launch dataset_example/extract_latent.py --droid_hf_path /cephfs/shared/droid_hf/droid_1.0.1 --droid_output_path dataset_example/droid_subset --svd_path /cephfs/shared/llm/stable-video-diffusion-img2vid --debug

