import json
import os
import random
import warnings
import traceback
import argparse
# from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import imageio
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange
from scipy.spatial.transform import Rotation as R  
import decord

from config import wm_orca_args

class Dataset_mix(Dataset):
    def __init__(
            self,
            args,
            mode = 'val',
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode

        # dataset stucture
        # dataset_root_path/dataset_name/annotation_name/mode/traj
        # dataset_root_path/dataset_name/video/mode/traj
        # dataset_root_path/dataset_name/latent_video/mode/traj

        # samples:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}

        # prepare all datasets path
        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []


        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split('+')
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split('+')
        self.prob = args.prob
        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            data_json_path = f'{dataset_meta_info_path}/{dataset_cfg}/{mode}_sample.json'
     
            with open(data_json_path, "r") as f:
                samples = json.load(f)
            dataset_path = [os.path.join(dataset_root_path, dataset_name) for sample in samples]
            print(f"ALL dataset, {len(samples)} samples in total")
            ids = np.random.choice(len(samples), args.max_num_samples, replace=False)
            samples = np.array(samples)[ids].tolist()
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))

            # prepare normalization
            with open(f'{dataset_meta_info_path}/{dataset_name}/stat.json', "r") as f:
                data_stat = json.load(f)
                state_p01 = np.array(data_stat['state_01'])[None,:]
                state_p99 = np.array(data_stat['state_99'])[None,:]
                self.norm_all.append((state_p01, state_p99))
        
        self.max_id = max(self.samples_len)
        print('samples_len:',self.samples_len, 'max_id:',self.max_id)

    def __len__(self):
        return self.max_id

    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path,'rb') as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids =  [int(frame_id) if frame_id < max_frames else max_frames-1 for frame_id in frame_ids]
        frame_data = video_tensor[frame_ids]
        return frame_data

    def _get_frames(self, label, frame_ids, cam_id, pre_encode, video_dir, use_img_cond=False):
        # directly load videos latent after svd-vae encoder
        assert cam_id is not None
        assert pre_encode == True
        if pre_encode: 
            video_path = label['latent_videos'][cam_id]['latent_video_path']
            video_path = os.path.join(video_dir,video_path)
            try:
                frames = self._load_latent_video(video_path, frame_ids)
            except:
                video_path = video_path.replace("latent_videos", "latent_videos_svd")
                frames = self._load_latent_video(video_path, frame_ids)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id = temp_cam_id, pre_encode = pre_encode, video_dir=video_dir)
        return frames, temp_cam_id

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def __getitem__(self, index, return_frame_ids=False):

        # first sample the dataset id, than sample the data from the dataset
        dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = dataset_path[index]

        # get annotation
        frame_ids = sample['frame_ids']
        ann_file = f'{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample["episode_id"]}.json'
        with open(ann_file, "r") as f:
            label = json.load(f)
            
        # since we downsample the video from 45hz to 5 hz to save the storage space, the frame id is 1/9 of the state id
        joint_len = len(label['observation.state.joint_position'])-1
        frame_len = np.floor(joint_len / self.args.down_sample)
        skip = random.randint(1, 2)
        skip_his = int(skip*4)
        p = random.random()
        if p < 0.15:
            skip_his = 0
        
        # rgb_id and state_id
        frame_now = frame_ids[0]
        rgb_id = []
        # getting the history rgb ids
        # using the max method to prevent negative values which would take future frames into frames
        for i in range(self.args.num_history,0,-1):
            rgb_id.append(int(frame_now - i*skip_his))
        rgb_id.append(frame_now)

        # getting the future rgb ids
        # the skip ensure temporal robustness => temporal augmentation
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i*skip))
        rgb_id = np.array(rgb_id)
        # ensuring we have no negative indices
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()
        rgb_id = [int(frame_id) for frame_id in rgb_id]
        state_id = np.array(rgb_id)*self.args.down_sample


        # prepare data
        data = dict()

        # instructions
        data['text'] = label['texts'][0]

        # stack tokens of multi-view
        # NOTE: Here we can easily define more or less views for condition
        if self.args.num_views == 1:
            cond_cam_id = np.random.randint(0, 1)
            latnt_cond,_ = self._get_obs(label, rgb_id, cond_cam_id, pre_encode=True, video_dir=dataset_dir)
            data['latent'] = latnt_cond.float()
        elif self.args.num_views == 2:
            cond_cam_id1 = np.random.randint(0, 1)
            cond_cam_id2 = 2 # wrist view
            latnt_cond1,_ = self._get_obs(label, rgb_id, cond_cam_id1, pre_encode=True, video_dir=dataset_dir)
            latnt_cond2,_ = self._get_obs(label, rgb_id, cond_cam_id2, pre_encode=True, video_dir=dataset_dir)
            latent = torch.zeros((self.args.num_frames+self.args.num_history, 4, 64, 32), dtype=torch.float32)
            latent[:,:,0:32] =  latnt_cond1
            latent[:,:,32:64] = latnt_cond2
            data['latent'] = latent.float()
        elif self.args.num_views == 3:
            cond_cam_id1 = 0
            cond_cam_id2 = 1
            cond_cam_id3 = 2
            latnt_cond1,_ = self._get_obs(label, rgb_id, cond_cam_id1, pre_encode=True, video_dir=dataset_dir)
            latnt_cond2,_ = self._get_obs(label, rgb_id, cond_cam_id2, pre_encode=True, video_dir=dataset_dir)
            latnt_cond3,_ = self._get_obs(label, rgb_id, cond_cam_id3, pre_encode=True, video_dir=dataset_dir)
            latent = torch.zeros((self.args.num_frames+self.args.num_history, 4, 96, 32), dtype=torch.float32)
            latent[:,:,0:32] =  latnt_cond1
            latent[:,:,32:64] = latnt_cond2
            latent[:,:,64:96] = latnt_cond3
            data['latent'] = latent.float()

        # prepare action cond data
        cartesian_pose = np.array(label['observation.state.cartesian_position'])[state_id]
        gripper_pose = np.array(label['observation.state.hand_joint_position'])[state_id]
        action = np.concatenate((cartesian_pose, gripper_pose), axis=-1)
        action = self.normalize_bound(action, state_p01, state_p99)
        data['action'] = torch.tensor(action).float()

        if return_frame_ids:
            return data, frame_ids
        return data
        

if __name__ == "__main__":

    from config import wm_orca_args
    args = wm_orca_args()
    train_dataset = Dataset_mix(args,mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    for data in tqdm(train_loader,total=len(train_loader)):
        print(data['ann_file'])

    