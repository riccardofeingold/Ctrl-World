import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ctrl_world import DinoV2VisualActionEncoder
import torch

def test_dino_visual_encoder():
    import time
    # load a sequence of RGB frames which represent the segmentation of the hand. (B, T, C, H, W)
    seg_sequence = torch.randn(1, 10, 1, 256, 256)
    # seg_sequence = seg_sequence.to("cuda:7")
    visual_action_encoder = DinoV2VisualActionEncoder(num_in_channels=1, embed_dim=1024)
    # visual_action_encoder.to("cuda:7")
    start_time = time.time()
    action_hidden = visual_action_encoder(seg_sequence)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(action_hidden.shape)
    return action_hidden

if __name__ == "__main__":
    action_hidden = test_dino_visual_encoder()