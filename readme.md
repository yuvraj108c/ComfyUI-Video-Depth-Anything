<div align="center">

# ComfyUI Video Depth Anything
<a href="https://arxiv.org/abs/2501.12375"><img src='https://img.shields.io/badge/arXiv-Video Depth Anything-red' alt='Paper PDF'></a>
<a href='https://videodepthanything.github.io'><img src='https://img.shields.io/badge/Project_Page-Video Depth Anything-green' alt='Project Page'></a>

This project is an unofficial ComfyUI implementation of [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything),  for depth estimation on long videos without compromising quality, consistency, or generalization ability.

![0126-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/2db00d84-9de3-434b-a36b-1981f4399e09)

</div>


## Installation

Navigate to the ComfyUI `/custom_nodes` directory

```bash
git clone https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git
cd ./ComfyUI-Video-Depth-Anything
pip install -r requirements.txt
```

Xformers (optional)
```bash
pip install xformers
```

## Usage
- Load [example workflow](workflows/video_depth_anything.json) 
- Models will download automatically to `ComfyUI/models/videodepthanything`

## Note
- The large model doesn't work with 24GB vram


## Citation

```bibtex
@article{video_depth_anything,
  title={Video Depth Anything: Consistent Depth Estimation for Super-Long Videos},
  author={Chen, Sili and Guo, Hengkai and Zhu, Shengnan and Zhang, Feihu and Huang, Zilong and Feng, Jiashi and Kang, Bingyi}
  journal={arXiv:2501.12375},
  year={2025}
}
```
## LICENSE
- Video-Depth-Anything-Small model is under the Apache-2.0 license. 
- Video-Depth-Anything-Large model is under the CC-BY-NC-4.0 license.