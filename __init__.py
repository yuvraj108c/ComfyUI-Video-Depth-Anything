from .node import LoadVideoDepthAnythingModel, VideoDepthAnythingProcess, VideoDepthAnythingOutput, VideoDepthAnythingSaveEXR
 
NODE_CLASS_MAPPINGS = { 
    "LoadVideoDepthAnythingModel" : LoadVideoDepthAnythingModel,
    "VideoDepthAnythingProcess" : VideoDepthAnythingProcess,
    "VideoDepthAnythingOutput" : VideoDepthAnythingOutput,
    "VideoDepthAnythingSaveEXR" : VideoDepthAnythingSaveEXR
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "LoadVideoDepthAnythingModel" : "Load Video Depth Anything Model",
     "VideoDepthAnythingProcess" : "Video Depth Anything Process",
     "VideoDepthAnythingOutput" : "Video Depth Anything Output Depth Map",
     "VideoDepthAnythingSaveEXR" : "Video Depth Anything Save EXR"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']