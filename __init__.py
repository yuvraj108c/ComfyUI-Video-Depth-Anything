from .node import LoadVideoDepthAnythingModel, VideoDepthAnythingProcess, VideoDepthAnythingOutput, VideoDepthAnythingSaveEXR, VideoDepthAnythingSaveContext, VideoDepthAnythingLoadContext

NODE_CLASS_MAPPINGS = {
    "LoadVideoDepthAnythingModel" : LoadVideoDepthAnythingModel,
    "VideoDepthAnythingProcess" : VideoDepthAnythingProcess,
    "VideoDepthAnythingOutput" : VideoDepthAnythingOutput,
    "VideoDepthAnythingSaveEXR" : VideoDepthAnythingSaveEXR,
    "VideoDepthAnythingSaveContext" : VideoDepthAnythingSaveContext,
    "VideoDepthAnythingLoadContext" : VideoDepthAnythingLoadContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoDepthAnythingModel" : "Load Video Depth Anything Model",
    "VideoDepthAnythingProcess" : "Video Depth Anything Process",
    "VideoDepthAnythingOutput" : "Video Depth Anything Output Depth Map",
    "VideoDepthAnythingSaveEXR" : "Video Depth Anything Save EXR",
    "VideoDepthAnythingSaveContext" : "Video Depth Anything Save Context",
    "VideoDepthAnythingLoadContext" : "Video Depth Anything Load Context",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
