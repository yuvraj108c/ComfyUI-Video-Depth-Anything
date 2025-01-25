import os
import torch
import numpy as np
import matplotlib.cm as cm
import cv2

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

from .video_depth_anything.video_depth import VideoDepthAnything

def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def preprocess(tensor, target_fps=-1, max_res=-1):
    # Convert from float [0,1] to uint8 [0,255]
    frames = (tensor.numpy() * 255).astype(np.uint8)
    
    original_height, original_width = frames.shape[1:3]
    
    # Resize if max_res specified
    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = ensure_even(round(original_height * scale))
        width = ensure_even(round(original_width * scale))
        
        resized_frames = []
        for frame in frames:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            resized_frames.append(frame)
        frames = np.array(resized_frames)
    
    # Default fps as 1:1 mapping if not specified
    fps = target_fps if target_fps > 0 else len(frames)
    
    return frames, fps

def postprocess_inferno(depths):
    colormap = np.array(cm.get_cmap("inferno").colors)
    d_min, d_max = depths.min(), depths.max()
    depth_frames = []
    for i in range(depths.shape[0]):
        depth = depths[i]
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
        depth_frames.append(depth_vis)
    return torch.from_numpy(np.array(depth_frames).astype(np.float32) / 255.0)

def postprocess_gray(depths):
    # Use the "gray" colormap directly
    colormap = cm.get_cmap("gray")  # Get the grayscale colormap
    d_min, d_max = depths.min(), depths.max()
    depth_frames = []
    
    for i in range(depths.shape[0]):
        depth = depths[i]
        # Normalize depth values to the range [0, 1]
        depth_norm = (depth - d_min) / (d_max - d_min)
        # Map normalized depth values to grayscale colormap
        depth_vis = (colormap(depth_norm)[:, :, :3] * 255).astype(np.uint8)  # Ignore alpha channel
        depth_frames.append(depth_vis)
    
    # Convert the list of frames to a tensor and normalize to [0, 1]
    return torch.from_numpy(np.array(depth_frames).astype(np.float32) / 255.0)

class LoadVideoDepthAnythingModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['video_depth_anything_vits.pth', 'video_depth_anything_vitl.pth'], {"default": 'video_depth_anything_vits.safetensors'}),
            },
        }

    RETURN_TYPES = ("VDAMODEL",)
    RETURN_NAMES = ("vda_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "VideoDepthAnything"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        download_path = os.path.join(folder_paths.models_dir, "videodepthanything")
        model_path = os.path.join(download_path, model)

        if not os.path.exists(model_path):
            print(f"[VideoDepthAnything] - Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download

            if model == "video_depth_anything_vits.pth":
                snapshot_download(repo_id="depth-anything/Video-Depth-Anything-Small", 
                                    allow_patterns=[f"*{model}*"],
                                    local_dir=download_path, 
                                    local_dir_use_symlinks=False)
            else:
                snapshot_download(repo_id="depth-anything/Video-Depth-Anything-Large", 
                                    allow_patterns=[f"*{model}*"],
                                    local_dir=download_path, 
                                    local_dir_use_symlinks=False)

        print(f"[VideoDepthAnything] - Loading model from: {model_path}")

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        
        }
        if "vits" in model:
            encoder = "vits"
        elif "vitl" in model:
            encoder = "vitl"

        self.model = VideoDepthAnything(**model_configs[encoder])
        state_dict = load_torch_file(model_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval()
        
        return (self.model,)

class VideoDepthAnythingProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vda_model": ("VDAMODEL", ),
                    "images": ("IMAGE", ),
                    "target_fps": ("FLOAT", {"default": 15}),
                    "input_size": ("INT",{"default": 518}),
                    "max_res": ("INT",{"default": 1280}),
                    "colormap": (['inferno', 'gray'], {"default": 'gray'}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES =("image", "fps", )
    FUNCTION = "process"
    CATEGORY = "VideoDepthAnything"

    def process(self, vda_model, images, target_fps, input_size, max_res, colormap):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        vda_model.to(device)
        pbar = ProgressBar(images.shape[0])
        images_np,fps = preprocess(images, target_fps, max_res)
        depths, fps = vda_model.infer_video_depth(images_np, target_fps=target_fps, input_size=input_size, device=device, pbar=pbar)

        vda_model.to(offload_device)

        if colormap == "inferno":
            output = postprocess_inferno(depths)
        else:
            output = postprocess_gray(depths)
        return (output,fps, )
