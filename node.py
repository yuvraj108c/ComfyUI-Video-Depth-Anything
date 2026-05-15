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

def preprocess(tensor, max_res=-1):
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
    
    return frames

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
    d_min, d_max = depths.min(), depths.max()
    
    # global normalisation
    depths = (depths - d_min) / (d_max - d_min + 1e-6)

    # to Torch
    tensor = torch.from_numpy(depths.astype(np.float32))  # [T,H,W]
    tensor = tensor.unsqueeze(-1).repeat(1, 1, 1, 3)      # [T,H,W,3]
    return tensor

class LoadVideoDepthAnythingModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['video_depth_anything_vits.pth', 'video_depth_anything_vitb.pth', 'video_depth_anything_vitl.pth', 'metric_video_depth_anything_vits.pth', 'metric_video_depth_anything_vitb.pth', 'metric_video_depth_anything_vitl.pth'], {"default": 'video_depth_anything_vits.safetensors'})
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

            match model:
                case "video_depth_anything_vits.pth":
                    repo_id = "depth-anything/Video-Depth-Anything-Small"
                case "video_depth_anything_vitb.pth":
                    repo_id = "depth-anything/Video-Depth-Anything-Base"
                case "video_depth_anything_vitl.pth":
                    repo_id = "depth-anything/Video-Depth-Anything-Large"
                case "metric_video_depth_anything_vits.pth":
                    repo_id = "depth-anything/Metric-Video-Depth-Anything-Small"
                case "metric_video_depth_anything_vitb.pth":
                    repo_id = "depth-anything/Metric-Video-Depth-Anything-Base"
                case "metric_video_depth_anything_vitl.pth":
                    repo_id = "depth-anything/Metric-Video-Depth-Anything-Large"

            snapshot_download(repo_id=repo_id, allow_patterns=[f"*{model}*"], local_dir=download_path)

        print(f"[VideoDepthAnything] - Loading model from: {model_path}")

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        if "vits" in model:
            encoder = "vits"
        elif "vitb" in model:
            encoder = "vitb"
        elif "vitl" in model:
            encoder = "vitl"

        self.model = VideoDepthAnything(**model_configs[encoder], metric="metric" in model)
        state_dict = load_torch_file(model_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device).eval()
        
        return (self.model,)

class VideoDepthAnythingProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vda_model": ("VDAMODEL", ),
                "images": ("IMAGE", ),
                "input_size": ("INT",{"default": 518}),
                "max_res": ("INT",{"default": 1280, "min": 0, "tooltip": "Maximum resolution for the longer edge. Set to 0 to disable resizing (use if you have already downsized the images externally)."}),
                "precision": (['fp16', 'fp32'], {"default": 'fp16'}),
            },
            "optional": {
                "prev_context": ("VDACONTEXT", {"tooltip": "Context tensor from a previous run to maintain temporal consistency at the split point."}),
            },
        }

    RETURN_TYPES = ("DEPTHS", "VDACONTEXT")
    RETURN_NAMES = ("depths", "context")
    FUNCTION = "process"
    CATEGORY = "VideoDepthAnything"

    def process(self, vda_model, images, input_size, max_res, precision, prev_context=None):
        from .video_depth_anything.video_depth import INFER_LEN
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        vda_model.to(device)

        images_np = preprocess(images, max_res)  # uint8 [N, H, W, 3]

        if prev_context is not None:
            # prev_context is float32 [INFER_LEN, H, W, 3] in [0,1] — convert back to uint8 and prepend
            context_np = (prev_context.numpy() * 255).astype(np.uint8)
            images_np_input = np.concatenate([context_np, images_np], axis=0)
            prepend_len = INFER_LEN
        else:
            images_np_input = images_np
            prepend_len = 0

        pbar = ProgressBar(images_np_input.shape[0])
        depths = vda_model.infer_video_depth(images_np_input, input_size=input_size, device=str(device), pbar=pbar, fp32=True if precision == 'fp32' else False)

        # Drop depths that correspond to the prepended context frames
        depths = depths[prepend_len:]

        # Save last INFER_LEN preprocessed frames as context for the next run
        context = torch.from_numpy(images_np[-INFER_LEN:].astype(np.float32) / 255.0)

        vda_model.to(offload_device)
        return (depths, context)

class VideoDepthAnythingSaveContext:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "context": ("VDACONTEXT", ),
                    "filename": ("STRING", {"default": "vda_context"}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save_context"
    CATEGORY = "VideoDepthAnything"
    OUTPUT_NODE = True

    def save_context(self, context, filename):
        out_dir = folder_paths.get_output_directory()
        stem = filename[:-3] if filename.lower().endswith(".pt") else filename
        path = os.path.join(out_dir, f"{stem}.pt")
        torch.save(context, path)
        print(f"[VideoDepthAnything] - Saved context to {path}")
        return (path,)


class VideoDepthAnythingLoadContext:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "path": ("STRING", {"default": ""}),
                    "force_reload": ("BOOLEAN", {"default": True, "tooltip": "Always reload the file from disk, bypassing ComfyUI's cache. Keep enabled when the file may have been overwritten since the last run."}),
                    "load_context": ("BOOLEAN", {"default": True, "tooltip": "If disabled, no context is loaded and the output is None — downstream Process node will run without prev_context."}),
            },
        }
    RETURN_TYPES = ("VDACONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "load_context"
    CATEGORY = "VideoDepthAnything"

    @classmethod
    def IS_CHANGED(cls, path, force_reload, load_context):
        if force_reload:
            return float("nan")
        return f"{path}|{load_context}"

    def load_context(self, path, force_reload, load_context):
        if not load_context:
            print("[VideoDepthAnything] - load_context disabled, returning None")
            return (None,)
        context = torch.load(path, map_location="cpu")
        print(f"[VideoDepthAnything] - Loaded context from {path}")
        return (context,)


class VideoDepthAnythingOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "depths": ("DEPTHS", ),
                    "colormap": (['inferno', 'gray'], {"default": 'gray'}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images", )
    FUNCTION = "output"
    CATEGORY = "VideoDepthAnything"

    def output(self, depths, colormap):
        if colormap == "inferno":
            output = postprocess_inferno(depths)
        else:
            output = postprocess_gray(depths)

        return (output, )

class VideoDepthAnythingSaveEXR:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "depths": ("DEPTHS", ),
                    "folder_name": ("STRING", {"default": 'exr-1', "required": True})
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES =("exr_path", )
    FUNCTION = "save_exr"
    CATEGORY = "VideoDepthAnything"
    OUTPUT_NODE = True

    def save_exr(self, depths, folder_name):
        depth_exr_dir = os.path.join(folder_paths.output_directory, folder_name)
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()

        print(f"[VideoDepthAnything] - Saved EXR to {depth_exr_dir}")

        return (depth_exr_dir,)