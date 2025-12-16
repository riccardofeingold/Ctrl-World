import torch
import torch.nn.functional as F
import numpy as np
import cv2
import mediapy
import matplotlib.pyplot as plt
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder

def downsample_mask(mask, target_height, target_width, weight):
    """
    Downsample a binary mask to the target height and width using nearest neighbor interpolation.
    
    Args:
        mask: Input binary mask as a numpy array of shape (H, W) or (1, H, W)
        target_height: Desired height after downsampling
        target_width: Desired width after downsampling
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    if torch.max(torch.tensor(mask)) > 1.0:
        mask = mask / 255.0
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
    downsampled_mask = F.interpolate(mask_tensor, size=(target_height, target_width), mode='nearest')
    downsampled_mask *= weight
    return downsampled_mask

def convert_to_image_format(tensor):
    """
    Convert a tensor to image format (H, W, 3) with values in [0, 255].
    
    Args:
        tensor: Input tensor of shape (C, H, W) with values in [-1, 1]
        
    Returns:
        Numpy array of shape (H, W, 3) with values in [0, 255]
    """
    tensor = (tensor / 2.0 + 0.5).clamp(0, 1) * 255  # Scale to [0, 1]
    tensor = tensor.permute(1, 2, 0)  # Change to (H, W, C)
    image = tensor.cpu().numpy().astype(np.uint8)
    return image

def main(svd_path, image_path, binary_mask_path, device):
    # decoding kwargs
    decode_kwargs = {
        "num_frames": 1,
    }

    # load VAE
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)
    print(f"Config scaling factor: {vae.config.scaling_factor}") 

    # read image
    image = mediapy.read_image(image_path) 
    image_tensor = torch.tensor(image).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    image_tensor = image_tensor[:3, :, :]  # ensure 3 channels
    image_tensor = image_tensor.unsqueeze(0)  # add batch dimension
    image_tensor = image_tensor.to(device)
    print(f"Image shape: {image_tensor.shape}")

    # read binary mask
    binary_image = mediapy.read_image(binary_mask_path)
    binary_image_tensor = torch.tensor(binary_image) / 255.0
    binary_image_tensor = binary_image_tensor.unsqueeze(0).unsqueeze(0)  # add batch dimension
    binary_image_tensor = binary_image_tensor.repeat(1, 3, 1, 1)  # repeat to have 3 channels
    binary_image_tensor = binary_image_tensor.to(device)
    print(f"Binary image tensor shape: {binary_image_tensor.shape}")
    print(f"Min: {binary_image_tensor.min()}, Max: {binary_image_tensor.max()}")

    # start testing hypotheses
    video_outputs = {
        "full_weight": [],
        "first_feature_map_weight": [],
        "second_feature_map_weight": [],
        "third_feature_map_weight": [],
        "fourth_feature_map_weight": [],
    }
    for weight in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 5.0, 10.0, 15.0, 30.0]:
        new_dir = f"{args.logging_dir}/weight_{weight}"
        import os
        os.makedirs(new_dir, exist_ok=True)
        print(f"Testing with weight: {weight}")
        with torch.no_grad():
            # weighting image in pixel space
            weighted_image = image_tensor * binary_image_tensor * weight + image_tensor * (1 - binary_image_tensor)
            cv2.imwrite(f"{new_dir}/weighted_image.png", convert_to_image_format(weighted_image[0]))

            # encode weighted image to latent space
            latent_weighted_image = vae.encode(weighted_image).latent_dist.sample().mul_(vae.config.scaling_factor)
            reconstructed_weighted_image = vae.decode(latent_weighted_image / vae.config.scaling_factor, **decode_kwargs).sample
            print(f"Min: {torch.min(reconstructed_weighted_image)}, Max: {torch.max(reconstructed_weighted_image)}")
            cv2.imwrite(f"{new_dir}/reconstructed_weighted_image.png", convert_to_image_format(reconstructed_weighted_image[0]))

            # encode original image
            latent_image = vae.encode(image_tensor).latent_dist.sample().mul_(vae.config.scaling_factor)
            # encode binary mask using VAE
            latent_binary = vae.encode(binary_image_tensor).latent_dist.sample().mul_(vae.config.scaling_factor)
            # downsample binary mask using nearest neighbor
            downsample_mask_tensor = downsample_mask(binary_image, latent_binary.shape[2], latent_binary.shape[3], weight).to(device)
            print(f"Downsampled mask tensor shape Min: {torch.min(downsample_mask_tensor)}, Max: {torch.max(downsample_mask_tensor)}")
            cv2.imwrite(f"{new_dir}/downsample_mask_tensor.png", ((downsample_mask_tensor[0].permute(1, 2, 0).cpu().numpy())).astype(np.uint8))

            # weighting in latent space using latent_binary from VAE
            weighted_in_latent = latent_image * latent_binary * weight + latent_image * (1 - latent_binary)
            print("Min and Max of weighted_in_latent:", torch.min(weighted_in_latent), torch.max(weighted_in_latent))
            # compute MSE in latent space
            mse_latent = torch.mean((latent_weighted_image - weighted_in_latent)**2)
            print(f"MSE in latent space: {mse_latent.item()}")
            # reconstructing the weighted_in_latent space to pixel space
            reconstructed_weighted_in_latent = vae.decode(weighted_in_latent / vae.config.scaling_factor, **decode_kwargs).sample
            mse_in_pixel_space = torch.mean((reconstructed_weighted_in_latent - weighted_image)**2)
            print(f"MSE in pixel space after reconstruction: {mse_in_pixel_space.item()}")
            cv2.imwrite(f"{new_dir}/reconstructed_weighted_in_latent.png", convert_to_image_format(reconstructed_weighted_in_latent[0]))

            # weighting in latent space using downsampled mask (nearest neighbor method)
            weighted_using_downsampled_mask = latent_image * downsample_mask_tensor + latent_image * (1 - downsample_mask_tensor / weight)
            mse_latent_downsampled = torch.mean((latent_weighted_image - weighted_using_downsampled_mask)**2)
            print(f"MSE in latent space using downsampled mask: {mse_latent_downsampled.item()}")

            # reconstructing the weighted_using_downsampled_mask to pixel space
            reconstructed_weighted_using_downsampled_mask = vae.decode(weighted_using_downsampled_mask / vae.config.scaling_factor, **decode_kwargs).sample
            video_outputs["full_weight"].append(convert_to_image_format(reconstructed_weighted_using_downsampled_mask[0])[np.newaxis, ...])
            mse_latent_downsampled_reconstructed = torch.mean((reconstructed_weighted_using_downsampled_mask - weighted_image)**2)
            print(f"MSE in pixel space after reconstruction (from downsampled mask): {mse_latent_downsampled_reconstructed.item()}")
            cv2.imwrite(f"{new_dir}/reconstructed_weighted_using_downsampled_mask.png", convert_to_image_format(reconstructed_weighted_using_downsampled_mask[0]))
            # applying the weight to only one feature map
            for i in range(4):
                weighted_using_downsampled_mask = latent_image.clone()
                weighted_using_downsampled_mask[0, i, :, :] = latent_image[0, i, :, :] * downsample_mask_tensor[0] + latent_image[0, i, :, :] * (1 - downsample_mask_tensor[0] / weight)
                reconstructed_weighted_using_downsampled_mask = vae.decode(weighted_using_downsampled_mask / vae.config.scaling_factor, **decode_kwargs).sample
                video_outputs[f"{['first','second','third','fourth'][i]}_feature_map_weight"].append(convert_to_image_format(reconstructed_weighted_using_downsampled_mask[0])[np.newaxis, ...])
                cv2.imwrite(f"{new_dir}/reconstructed_weighted_using_downsampled_mask_frame_{i}.png", convert_to_image_format(reconstructed_weighted_using_downsampled_mask[0]))

            # reconstruct original latent image
            reconstruct_latent_image = vae.decode(latent_image / vae.config.scaling_factor, **decode_kwargs).sample
            cv2.imwrite(f"{new_dir}/reconstruct_latent_image.png", convert_to_image_format(reconstruct_latent_image[0]))
        
    # save as video
    for key in video_outputs:
        video_path = f"{args.logging_dir}/reconstructed_{key}_video.mp4"
        video_array = np.concatenate(video_outputs[key])
        mediapy.write_video(video_path, video_array, fps=2)
        print(f"Saved video to {video_path}")

def get_image_with_mask(image_path):
    """
    Load an image and let user draw a binary mask on it.
    
    Instructions:
    - Left click and drag to draw (white regions in mask)
    - Press 'r' to reset the mask
    - Press 'c' to clear the mask
    - Press 'q' or ESC to finish and return the mask
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Binary mask as numpy array (same height/width as image)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy for display and initialize mask
    display_image = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Drawing parameters
    drawing = False
    brush_size = 10
    
    def draw_circle(event, x, y, flags, param):
        nonlocal drawing, display_image, mask
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, (x, y), brush_size, 255, -1)
            cv2.circle(display_image, (x, y), brush_size, (0, 255, 0), -1)
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(mask, (x, y), brush_size, 255, -1)
            cv2.circle(display_image, (x, y), brush_size, (0, 255, 0), -1)
            
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    # Create window and set mouse callback
    window_name = 'Draw Mask (q to finish, r to reset, c to clear)'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_circle)
    
    print("Instructions:")
    print("- Left click and drag to draw")
    print("- Press '+' to increase brush size")
    print("- Press '-' to decrease brush size")
    print("- Press 'r' to reset (reload original image)")
    print("- Press 'c' to clear mask")
    print("- Press 'q' or ESC to finish")
    
    while True:
        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('r'):  # Reset
            display_image = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        elif key == ord('c'):  # Clear
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            display_image = image.copy()
        elif key == ord('+') or key == ord('='):  # Increase brush size
            brush_size = min(brush_size + 2, 50)
            print(f"Brush size: {brush_size}")
        elif key == ord('-') or key == ord('_'):  # Decrease brush size
            brush_size = max(brush_size - 2, 1)
            print(f"Brush size: {brush_size}")
    
    cv2.destroyAllWindows()
    
    # Return binary mask (0 or 1)
    return (mask > 0).astype(np.uint8)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--binary_mask_path', type=str, default="tests/test_example_object_hand_centric/mask.png")
    parser.add_argument('--image_path', type=str, default="tests/test_example_object_hand_centric/orca_image_example.png")
    parser.add_argument('--svd_path', type=str, default='stabilityai/stable-video-diffusion-img2vid')
    parser.add_argument('--logging_dir', type=str, default='tests/test_example_object_hand_centric/logs')
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.binary_mask_path is None:
        mask = get_image_with_mask(args.image_path)
        
        # Display the final mask
        cv2.imshow("Final Binary Mask", mask * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the mask if needed
        cv2.imwrite("mask.png", mask * 255)

    accelerator = Accelerator()
    main(args.svd_path, args.image_path, args.binary_mask_path, accelerator.device)