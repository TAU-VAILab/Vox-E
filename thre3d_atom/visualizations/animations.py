from typing import Sequence, Optional
import imageio

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from thre3d_atom.thre3d_reprs.cross_attn import text_under_image
from thre3d_atom.modules.sds_trainer import _get_dir_batch_from_poses
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS, NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import (
    CameraPose,
    CameraIntrinsics,
    scale_camera_intrinsics,
    postprocess_depth_map,
    to8b,
)
from thre3d_atom.utils.logging import log


def render_camera_path_for_volumetric_model(
    vol_mod: VolumetricModel,
    camera_path: Sequence[CameraPose],
    camera_intrinsics: CameraIntrinsics,
    render_scale_factor: Optional[float] = None,
    overridden_num_samples_per_ray: Optional[int] = None,
    image_save_freq: Optional[int] = None,
    image_save_path: Optional[str] = None,
) -> np.array:
    if render_scale_factor is not None:
        # Render downsampled images for speed if requested
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics, render_scale_factor
        )

    overridden_config_dict = {}
    if overridden_num_samples_per_ray is not None:
        overridden_config_dict.update(
            {"num_samples_per_ray": overridden_num_samples_per_ray}
        )

    rendered_frames = []
    total_frames = len(camera_path) + 1
    for frame_num, render_pose in enumerate(camera_path):
        log.info(f"rendering frame number: ({frame_num + 1}/{total_frames})")
        rendered_output = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame = rendered_output.colour.numpy()
        depth_frame = rendered_output.depth.numpy()
        acc_frame = rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].numpy()

        # apply post-processing to the depth frame
        colour_frame = to8b(colour_frame)
        depth_frame = postprocess_depth_map(depth_frame, acc_map=acc_frame)
        # tile the acc_frame to have 3 channels
        # also invert it for a better visualization
        acc_frame = to8b(1.0 - np.tile(acc_frame, (1, 1, NUM_COLOUR_CHANNELS)))

        frame = np.concatenate([colour_frame, depth_frame, acc_frame], axis=1)
        rendered_frames.append(frame)

        # save image if necessary (used for plots and stuff)
        if image_save_freq != None:
            if frame_num % image_save_freq == 0:
                imageio.imwrite(
                image_save_path / f"{frame_num}.png",
                colour_frame,
        )

    return np.stack(rendered_frames)

def render_camera_path_for_volumetric_model_attn(
        vol_mod: VolumetricModel,
        camera_path: Sequence[CameraPose],
        camera_intrinsics: CameraIntrinsics,
        render_scale_factor: Optional[float] = None,
        overridden_num_samples_per_ray: Optional[int] = None
) -> np.array:
    if render_scale_factor is not None:
        # Render downsampled images for speed if requested
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics, render_scale_factor
        )

    overridden_config_dict = {}
    if overridden_num_samples_per_ray is not None:
        overridden_config_dict.update(
            {"num_samples_per_ray": overridden_num_samples_per_ray}
        )

    rendered_frames, attn_frames = [], []
    total_frames = len(camera_path) + 1
    for frame_num, render_pose in enumerate(camera_path):
        log.info(f"rendering frame number: ({frame_num + 1}/{total_frames})")
        rendered_output = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        rendered_attn = vol_mod.render_attn(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame = rendered_output.colour.numpy()
        depth_frame = rendered_output.depth.numpy()
        acc_frame = rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].numpy()

        attn_frame = rendered_attn.attn.squeeze(-1).cpu().numpy()

        # apply post-processing to the depth frame
        colour_frame = to8b(colour_frame)
        depth_frame = postprocess_depth_map(depth_frame, acc_map=acc_frame)
        # tile the acc_frame to have 3 channels
        # also invert it for a better visualization
        cmp = cm.get_cmap('jet')
        norm = colors.Normalize(vmin=np.min(attn_frame), vmax=np.max(attn_frame))
        attn_frame = cmp(norm(attn_frame))[:, :, :3]
        attn_frame_save = (0.5 * attn_frame) + (0.5 * rendered_output.colour.numpy())
        attn_frame = to8b(attn_frame)
        attn_frame_save = to8b(attn_frame_save)
        frame = np.concatenate([colour_frame, depth_frame, attn_frame], axis=1)
        rendered_frames.append(frame)
        attn_frames.append(attn_frame_save)


    return np.stack(rendered_frames), attn_frames


def render_camera_path_for_volumetric_model_attn_blend(
        vol_mod: VolumetricModel,
        camera_path: Sequence[CameraPose],
        camera_intrinsics: CameraIntrinsics,
        render_scale_factor: Optional[float] = None,
        overridden_num_samples_per_ray: Optional[int] = None
) -> np.array:
    if render_scale_factor is not None:
        # Render downsampled images for speed if requested
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics, render_scale_factor
        )

    overridden_config_dict = {}
    if overridden_num_samples_per_ray is not None:
        overridden_config_dict.update(
            {"num_samples_per_ray": overridden_num_samples_per_ray}
        )

    rendered_frames, attn_frames = [], []
    total_frames = len(camera_path) + 1
    for frame_num, render_pose in enumerate(camera_path):
        log.info(f"rendering frame number: ({frame_num + 1}/{total_frames})")
        rendered_output = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        rendered_attn = vol_mod.render_attn(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame = rendered_output.colour.numpy()
        depth_frame = rendered_output.depth.numpy()
        acc_frame = rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].numpy()

        attn_frame = rendered_attn.attn.squeeze(-1).cpu().numpy()
        #attn_frame = 1-attn_frame


        # apply post-processing to the depth frame
        colour_frame = to8b(colour_frame)
        depth_frame = postprocess_depth_map(depth_frame, acc_map=acc_frame)
        # tile the acc_frame to have 3 channels
        # also invert it for a better visualization

        def shift_cmap(cmap, frac):
            """Shifts a colormap by a certain fraction.
            Keyword arguments:
            cmap -- the colormap to be shifted. Can be a colormap name or a Colormap object
            frac -- the fraction of the colorbar by which to shift (must be between 0 and 1)
            """
            N = 256
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            n = cmap.name
            x = np.linspace(0, 1, N)
            out = np.roll(x, int(N * frac))
            new_cmap = colors.LinearSegmentedColormap.from_list(f'{n}_s', cmap(out))
            return new_cmap
        
        cmp = cm.get_cmap('jet')
        #cmp = shift_cmap(cmp, 0.5)
        norm = colors.Normalize(vmin=np.min(attn_frame), vmax=np.max(attn_frame))
        attn_frame = cmp(norm(attn_frame))[:, :, :3]
        attn_frame_save = (0.5 * attn_frame) + (0.5 * rendered_output.colour.numpy())
        attn_frame = to8b(attn_frame)
        attn_frame_save = to8b(attn_frame_save)
        frame = np.concatenate([colour_frame, depth_frame, attn_frame], axis=1)
        rendered_frames.append(frame)
        attn_frames.append(attn_frame_save)


    return np.stack(rendered_frames), attn_frames

def render_camera_path_for_volumetric_model_attn(
        vol_mod: VolumetricModel,
        camera_path: Sequence[CameraPose],
        camera_intrinsics: CameraIntrinsics,
        render_scale_factor: Optional[float] = None,
        overridden_num_samples_per_ray: Optional[int] = None,
         timestamp=0
) -> np.array:
    if render_scale_factor is not None:
        # Render downsampled images for speed if requested
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics, render_scale_factor
        )

    overridden_config_dict = {}
    if overridden_num_samples_per_ray is not None:
        overridden_config_dict.update(
            {"num_samples_per_ray": overridden_num_samples_per_ray}
        )

    rendered_frames_n = []
    attn = []
    mmax = 0
    total_frames = len(camera_path) + 1
    for frame_num, render_pose in enumerate(camera_path):
        log.info(f"rendering frame number: ({frame_num + 1}/{total_frames})")
        rendered_output = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame = rendered_output.colour.numpy()
        out_imgs = rendered_output.colour.unsqueeze(0)
        out_imgs = out_imgs.permute((0, 3, 1, 2))
        depth_frame = rendered_output.depth.numpy()
        acc_frame = rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].numpy()

        # apply post-processing to the depth frame
        colour_frame = to8b(colour_frame)
        depth_frame = postprocess_depth_map(depth_frame, acc_map=acc_frame)
        #dir = _get_dir_batch_from_poses(torch.from_numpy(
        #    np.hstack((render_pose.rotation, render_pose.translation))
        #).unsqueeze(0).to(device))[0]
        #m_prompt = prompt + f", {dir} view"
        #acc_frame, t = sd_model.get_attn_map(prompt=[m_prompt], pred_rgb=out_imgs, timestamp=timestamp,
        #                                     indices_to_alter=index_to_attn)
        mmax = np.amax(acc_frame) if np.amax(acc_frame) > mmax else mmax
        attn.append([rendered_output.colour.numpy(), acc_frame])

        # create grand concatenated frame horizontally
        frame = np.concatenate([colour_frame, depth_frame], axis=1)
        rendered_frames_n.append(frame)

    f_attn, rendered_frames = [], []
    for i, l in enumerate(attn):
        out_im, attn = l
        cmp = cm.get_cmap('jet')
        norm = colors.Normalize(vmin=0, vmax=mmax.item())
        acc_frame = cmp(norm(attn))[:, :, :3]
        acc_frame = (0.5 * acc_frame) + (0.5 * out_im)
        acc_frame = to8b(acc_frame)
        frame = np.concatenate([rendered_frames_n[i], acc_frame], axis=1)
        rendered_frames.append(frame)

    return np.stack(rendered_frames), f_attn
