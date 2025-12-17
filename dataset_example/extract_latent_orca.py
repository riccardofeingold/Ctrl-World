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
    def __init__(self, old_path, new_path, svd_path, device, size=(192, 320), rgb_skip=3):
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
                    f'{self.old_path}/videos/{idx}/0_rgb.mp4',
                    f'{self.old_path}/videos/{idx}/1_rgb.mp4',
                    f'{self.old_path}/videos/{idx}/2_rgb.mp4']
        traj_info = {'success': success,
                     'observation.state.cartesian_position': obs_car,
                     'observation.state.joint_position': obs_joint,
                     'observation.state.hand_joint_position': obs_hand_joint,
                     'action.cartesian_position': action_car,
                     'action.hand_joint_position': action_hand_joint,
                    }
        

        # if f"{save_root}/videos/{data_type}/{traj_id}" exist, skip this trajectory
        try:
            self.process_traj(video_paths, traj_info, instruction, self.new_path, traj_id=traj_id, data_type=data_type, size=self.size, rgb_skip=self.skip, device=self.vae.device)
        except Exception as e:
            print(f"Error processing trajectory {traj_id}, skipping... Error: {e}")
            return 0
    
        return 0


    def process_traj(self, video_paths, traj_info, instruction, save_root,traj_id=0,data_type='val', size=(192,320), rgb_skip=3, device='cuda'):
        for video_id, video_path in enumerate(video_paths):
            # load and resize video and save
            video = mediapy.read_video(video_path)
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
            frames = frames[::rgb_skip]  # Skip frames to save memory here!!!
            x = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{save_root}/videos/{data_type}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{save_root}/videos/{data_type}/{traj_id}/{video_id}.mp4", resize_video, fps=5)

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
        
        # record cartesain aligned with video frames
        cartesian_pose = np.array(traj_info['observation.state.cartesian_position'])
        cartesian_hand_joint = np.array(traj_info['observation.state.hand_joint_position'])
        cartesian_states = np.concatenate((cartesian_pose, cartesian_hand_joint),axis=-1)[::rgb_skip].tolist()
        assert len(cartesian_states) == len(frames), f"Length mismatch: {len(cartesian_states)} vs {len(frames)}"  
        
        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info['success']),
            "video_length": frames.shape[0],
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
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--frame_size', type=parse_tuple, default=(256, 256))
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    wm_arguments = wm_orca_args()

    accelerator = Accelerator()
    rgb_skip = int(wm_arguments.original_fps / args.fps)
    print(f"Actual FPS after skipping: {wm_arguments.original_fps / rgb_skip} Hz")
    dataset = EncodeLatentDataset(
        old_path=args.orca_dataset_path,
        new_path= args.orca_output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=args.frame_size,
        rgb_skip=rgb_skip, # NOTE: Synthetic data is recorded at 45 fps but the decimation is set to 2 which means the fps is actually 45, we use 5 fps
    )
    tmp_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
        )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    for idx, _ in enumerate(tmp_data_loader):
        if idx == 5 and args.debug:
            break
        if idx % 100 == 0 and accelerator.is_main_process:
            print(f"Precomputed {idx} samples")

# accelerate launch dataset_example/extract_latent.py --droid_hf_path /cephfs/shared/droid_hf/droid_1.0.1 --droid_output_path dataset_example/droid_subset --svd_path /cephfs/shared/llm/stable-video-diffusion-img2vid --debug

