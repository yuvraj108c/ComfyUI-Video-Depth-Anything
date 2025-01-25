from .node import LoadVideoDepthAnythingModel, VideoDepthAnythingProcess
 
NODE_CLASS_MAPPINGS = { 
    "LoadVideoDepthAnythingModel" : LoadVideoDepthAnythingModel,
    "VideoDepthAnythingProcess" : VideoDepthAnythingProcess
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "LoadVideoDepthAnythingModel" : "Load Video Depth Anything Model",
     "VideoDepthAnythingProcess" : "Video Depth Anything Process"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']