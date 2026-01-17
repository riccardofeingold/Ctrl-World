# implement a function to read all pt files in a directory and return a list of tensors

import torch
import os

def read_pt_files(directory):
    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    tensors = [torch.load(os.path.join(directory, f)) for f in pt_files]
    print(f"Read {len(tensors)} tensors from {directory}")
    print(f"First tensor shape: {tensors[0].shape}")
    return tensors

if __name__ == "__main__":
    directory = {
        "latent_segmentation_videos": "datasets/2026-01-16T15-34-02/data_working_hand_mask_sine_hand_motions_10fps/latent_segmentation_videos/train/32",
        "latent_videos": "datasets/2026-01-16T15-34-02/data_working_hand_mask_sine_hand_motions_10fps/latent_videos/train/31"
    }
    for key, value in directory.items():
        tensors = read_pt_files(value)