"""RoboTwin 3-camera mosaic, shared by the latent precompute and simulator inference.

    +---------------------------+
    |   head camera [256, 320]  |
    +-------------+-------------+
    | left 128x160| right 128x160|
    +-------------+-------------+   -> [384, 320]

Must stay identical to the "robotwin" branch of RobotVideoDataset._get.
"""

import torch
import torchvision.transforms.functional as transforms_F

HEAD_SIZE = [256, 320]
WRIST_SIZE = [128, 160]
MOSAIC_SIZE = [384, 320]


def concat_cameras_robotwin(pixels: torch.Tensor) -> torch.Tensor:
    """[3 (head, left, right), T, C, H, W] -> [T, C, 384, 320]."""
    num_cam = pixels.shape[0]
    if num_cam != 3:
        raise ValueError(
            f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cam}"
        )

    def _resize(x, size):
        return transforms_F.resize(
            x,
            size=size,
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )

    cam_top = _resize(pixels[0], HEAD_SIZE)      # [T, C, 256, 320]
    cam_left = _resize(pixels[1], WRIST_SIZE)    # [T, C, 128, 160]
    cam_right = _resize(pixels[2], WRIST_SIZE)   # [T, C, 128, 160]
    bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T, C, 128, 320]
    return torch.cat([cam_top, bottom], dim=-2)        # [T, C, 384, 320]


def build_mosaic_from_frames(
    cam_frames: torch.Tensor,
    img_transforms,
    resize_t,
    crop_t,
    norm_t,
) -> torch.Tensor:
    """uint8 [3, T, C, H, W] -> [-1, 1] mosaic [T, C, 384, 320], using the training transforms."""
    cams = []
    for ci in range(cam_frames.shape[0]):
        x = cam_frames[ci]
        for tr in img_transforms:
            x = tr(x)
        cams.append(x)
    video = concat_cameras_robotwin(torch.stack(cams, dim=0))
    return norm_t(crop_t(resize_t(video)))
