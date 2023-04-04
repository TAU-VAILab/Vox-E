import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import imageio
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
import wandb

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.data.utils import infinite_dataloader
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.rendering.volumetric.utils.misc import (
    cast_rays,
    flatten_rays,
)

from thre3d_atom.thre3d_reprs.renderers import render_sh_voxel_grid_attn
from thre3d_atom.thre3d_reprs.voxels import (
    VoxelGrid,
)
from thre3d_atom.thre3d_reprs.sd import StableDiffusion
from thre3d_atom.utils.constants import (
    CAMERA_BOUNDS,
    CAMERA_INTRINSICS,
    HEMISPHERICAL_RADIUS,
)
from thre3d_atom.utils.imaging_utils import CameraPose

# All the TrainProcedures below follow this function-type
from thre3d_atom.utils.logging import log
from thre3d_atom.visualizations.static import (
    visualize_sh_vox_grid_vol_mod_rendered_feedback,
    visualize_sh_vox_grid_vol_mod_rendered_feedback_attn,
)

from thre3d_atom.modules.refinement_functions import (
    visualize_and_log_attention_maps,
    calc_loss_on_attn_grid,
    get_edit_region,
    log_and_vis_render_diff
)

from thre3d_atom.utils.imaging_utils import (
    get_random_pose,
)

HEMISPHERICAL_RADIUS_CONSTANT = 4.0311

dir_to_num_dict = {'side': 0, 'overhead': 1, 'back': 2, 'front': 3}
mse_loss = torch.nn.MSELoss(reduction='none')


# TrainProcedure = Callable[[VolumetricModel, Dataset, ...], VolumetricModel]


def refine_edited_relu_field(
        vol_mod_edit: VolumetricModel,
        vol_mod_object: VolumetricModel,
        vol_mod_output: VolumetricModel,
        vol_mod_ref: VolumetricModel,
        train_dataset: PosedImagesDataset,
        hf_auth_token: str,
        # required arguments:
        output_dir: Path,
        prompt: str,
        edit_idx: int,
        timestamp: int,
        image_dims: tuple,
        object_idx: int = None,
        num_iterations: int = 2000,
        # learning_rate and related arguments
        learning_rate: float = 0.03,
        lr_decay_gamma_per_stage: float = 0.1,
        lr_decay_steps_per_stage: int = 2000,
        # option to have a specific feedback_pose_for_visual feedback rendering
        render_feedback_pose: Optional[CameraPose] = None,
        # various training-loop frequencies
        save_freq: int = 1000,
        feedback_freq: int = 100,
        summary_freq: int = 10,
        # regularization option:
        apply_diffuse_render_regularization: bool = False,
        # miscellaneous options can be left untouched
        verbose_rendering: bool = True,
        attn_tv_weight: float = 0.001,
        kval: float = 5.0,
        edit_mask_thresh: float = 0.992,
        num_obj_voxels_thresh: int = 5000,
        min_num_edit_voxels: int = 300,
        top_k_edit_thresh: int = 300,
        top_k_obj_thresh: int = 200,
        log_wandb: bool = False,
) -> VolumetricModel:
    """
    ------------------------------------------------------------------------------------------------------
    |                               !!! :D LONG FUNCTION ALERT :D !!!                                    |
    ------------------------------------------------------------------------------------------------------
    trains a volumetric model given a dataset of images and corresponding poses
    Args:
        vol_mod: the volumetricModel to be trained with this procedure. Please note that it should have
                 an sh-based VoxelGrid as its underlying thre3d_repr.
        train_dataset: PosedImagesDataset used for training
        output_dir: path to the output directory where the assets of the training are to be written
        random_initializer: the pytorch initialization routine used for features of voxel_grid
        test_dataset: optional dataset of test images and poses :)
        image_batch_cache_size: batch of images from which rays are sampled per training iteration
        ray_batch_size: number of randomly sampled rays used per training iteration
        num_iterations_per_stage: iterations performed per stage
        scale_factor: factor by which the grid is up-scaled after each stage
        learning_rate: learning rate used for differential optimization
        lr_decay_gamma_per_stage: value of gamma for learning rate-decay in a single stage
        lr_decay_steps_per_stage: steps after which exponential learning rate decay is kicked in
        stagewise_lr_decay_gamma: gamma reduction of learning rate after each stage
        render_feedback_pose: optional feedback pose used for generating the rendered feedback
        save_freq: number of iterations after which checkpoints are saved
        test_freq: number of iterations after which testing scores are computed
        feedback_freq: number of iterations after which feedback is generated
        summary_freq: number of iterations after which current loss is logged to console
        apply_diffuse_render_regularization: whether to apply the diffuse render regularization
        verbose_rendering: bool to control whether to show verbose details while generating rendered feedback
        fast_debug_mode: bool to control fast_debug_mode, skips testing and some other things
        diffuse_weight: weight for diffuse loss - used for regularization
        spcular_weight: weight for specular loss - used for regularization

    Returns: the trained version of the VolumetricModel. Also writes multiple assets to disk
    """
    # assertions about the VolumetricModel being used with this TrainProcedure :)
    assert isinstance(vol_mod_edit.thre3d_repr, VoxelGrid), (
        f"sorry, cannot use a {type(vol_mod_edit.thre3d_repr)} with this TrainProcedure :(; "
        f"only a {type(VoxelGrid)} can be used"
    )
    assert (
            vol_mod_edit._render_procedure_attn == render_sh_voxel_grid_attn
    ), f"sorry, non SH-based VoxelGrids cannot be used with this TrainProcedure"

    assert (
            prompt != "none"
    ), f"sorry, you have to supply a text prompt to use SDS"

    # init sds loss class
    sd_model = StableDiffusion(vol_mod_edit.device, "1.4", auth_token=hf_auth_token)
    im_h, im_w = image_dims
    direction_batch = None

    # save the real_feedback_test_image if it exists:
    feedback_pose_given = False
    if render_feedback_pose is not None:
        feedback_pose_given = True

    # setup output directories
    # fmt: off
    model_dir = output_dir / "saved_models"
    logs_dir = output_dir / "training_logs"
    tensorboard_dir = logs_dir / "tensorboard"
    render_dir = logs_dir / "rendered_output"
    for directory in (model_dir, logs_dir, tensorboard_dir,
                      render_dir):
        directory.mkdir(exist_ok=True, parents=True)
    # fmt: on

    # extract the camera_bounds and camera_intrinsics for rest of the procedure
    camera_bounds, camera_intrinsics = (
        train_dataset.camera_bounds,
        train_dataset.camera_intrinsics,
    )

    # start actual training
    log.info("beginning training")
    time_spent_actually_training = 0

    # -----------------------------------------------------------------------------------------
    #  Main Training Loop                                                                     |
    # -----------------------------------------------------------------------------------------

    # set optimizer edit
    params_edit = [{"params": vol_mod_edit.thre3d_repr.attn, "lr": learning_rate}]

    optimizer_edit = torch.optim.Adam(
        params=params_edit,
        betas=(0.9, 0.999),
    )

    # setup learning rate schedulers for the optimizer
    lr_scheduler_edit = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_edit, gamma=lr_decay_gamma_per_stage
    )

    # set optimizer object
    params_object = [{"params": vol_mod_object.thre3d_repr.attn, "lr": learning_rate}]

    optimizer_object = torch.optim.Adam(
        params=params_object,
        betas=(0.9, 0.999),
    )

    # display logs related to this training stage:
    log.info(
        f"voxel grid resolution: {vol_mod_edit.thre3d_repr.grid_dims} "
        f"training images resolution: [{im_h} x {im_w}]"
    )
    current_stage_lrs = [
        param_group["lr"] for param_group in optimizer_edit.param_groups
    ]
    log_string = f"current stage learning rates: {current_stage_lrs} "
    log.info(log_string)
    last_time = time.perf_counter()

    # -------------------------------------------------------------------------------------
    #  Single Stage Training Loop                                                         |
    # -------------------------------------------------------------------------------------

    for global_step in range(1, num_iterations + 1):
        # ---------------------------------------------------------------------------------
        #  Main Operations Performed Per Iteration                                        |
        # ---------------------------------------------------------------------------------
        # sample a batch rays and pixels for a single iteration
        # load a batch of images and poses (These could already be cached on GPU)
        # please check the `data.datasets` module

        total_loss_edit = 0
        total_loss_object = 0

        # -------------------
        #  Get Input Pose:  |
        # -------------------

        pose, dir = get_random_pose(HEMISPHERICAL_RADIUS_CONSTANT)
        unflattened_rays = cast_rays(
                    train_dataset.camera_intrinsics,
                    pose,
                    device=vol_mod_edit.device,
                )
        rays_batch = flatten_rays(unflattened_rays)
        direction_batch = [dir]

        # ----------------------
        #  Render RGB Output:  |
        # ----------------------

        rendered_output = vol_mod_edit.render(
            pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=False,
        )

        # -----------------------
        #  Get Attention Maps:  |
        # -----------------------

        out_imgs = rendered_output.colour.unsqueeze(0)
        out_imgs = out_imgs.permute((0, 3, 1, 2)).to(vol_mod_edit.device)
        m_prompt = prompt + f", {direction_batch[0]} view"

        # if no object idx is given (default) take the maximum between all non-edit tokens
        if object_idx == None:
            indices_to_fetch = list(range(1, edit_idx + 1))
        else:
            indices_to_fetch = [edit_idx, object_idx]

        gt, _ = sd_model.get_attn_map(prompt=m_prompt, pred_rgb=out_imgs, timestamp=timestamp,
                                      indices_to_fetch=indices_to_fetch)

        if object_idx == None:
            edit_attn_map = gt.pop(edit_idx - 1)
            rest_of_attn_maps = [t.unsqueeze(dim=-1) for t in gt]
            object_attn_map = torch.cat(rest_of_attn_maps, dim=-1)
            object_attn_map, _ = torch.max(object_attn_map, dim=-1)
            object_attn_map = object_attn_map.squeeze()
        else:
            edit_attn_map = gt[0]
            object_attn_map = gt[1]

        # -------------------------------------
        #  Get Attention Outputs from grids:  |
        # -------------------------------------

        edit_attn_rendered_batch = vol_mod_edit.render_rays_attn(rays_batch)
        edit_attn_rendered_batch = edit_attn_rendered_batch.attn

        object_attn_rendered_batch = vol_mod_object.render_rays_attn(rays_batch)
        object_attn_rendered_batch = object_attn_rendered_batch.attn

        # -----------------
        #  Incur Losses:  |
        # -----------------

        edit_attn_loss = calc_loss_on_attn_grid(attn_render=edit_attn_rendered_batch,
                                                attn_map=edit_attn_map,
                                                token="edit",
                                                global_step=global_step)

        object_attn_loss = calc_loss_on_attn_grid(attn_render=object_attn_rendered_batch,
                                                  attn_map=object_attn_map,
                                                  token="object",
                                                  global_step=global_step)

        edit_attn_render = edit_attn_rendered_batch.reshape(edit_attn_map.shape)
        object_attn_render = object_attn_rendered_batch.reshape(edit_attn_map.shape)

        total_loss_edit = total_loss_edit + edit_attn_loss
        tv_loss_edit =_tv_loss_on_grid(vol_mod_edit.thre3d_repr.attn)
        total_loss_edit = total_loss_edit + tv_loss_edit * attn_tv_weight

        total_loss_object = total_loss_object + object_attn_loss
        tv_loss_object =_tv_loss_on_grid(vol_mod_object.thre3d_repr.attn)
        total_loss_object = total_loss_object + tv_loss_object * attn_tv_weight

        # -------------
        #  Optimize:  |
        # -------------

        total_loss_edit.backward()
        optimizer_edit.step()
        optimizer_edit.zero_grad()

        total_loss_object.backward()
        optimizer_object.step()
        optimizer_object.zero_grad()

        # ------------
        #  Logging:  |
        # ------------

        if log_wandb:
            # Get attention Maps
            log_step = global_step + + num_iterations
            if (
                global_step % summary_freq == 0
                or global_step == 1
                or global_step == num_iterations
            ):
                visualize_and_log_attention_maps(gt, log_step)
                log_and_vis_render_diff(edit_attn_render, object_attn_render, log_step)
            wandb.log({"attn_loss_edit": edit_attn_loss}, step=log_step)
            wandb.log({"tv_loss_edit": tv_loss_edit}, step=log_step)
            wandb.log({"total_loss_edit": total_loss_edit}, step=log_step)
            wandb.log({"attn_loss_object": object_attn_loss}, step=log_step)
            wandb.log({"tv_loss_object": tv_loss_object}, step=log_step)
            wandb.log({"total_loss_object": total_loss_object}, step=log_step)
            wandb.log({"Input Direction": dir}, step=log_step)
            # ---------------------------------------------------------------------------------

        # rest of the code per iteration is related to saving/logging/feedback/testing
        time_spent_actually_training += time.perf_counter() - last_time

        # console loss feedback
        if (
                global_step % summary_freq == 0
                or global_step == 1
                or global_step == num_iterations
        ):
            loss_info_string = (
                f"Global Iteration: {global_step} "
                f"attn_loss: {edit_attn_loss.item(): .3f} "
            )
            log.info(loss_info_string)

        # step the learning rate schedulers
        if global_step % lr_decay_steps_per_stage == 0:
            lr_scheduler_edit.step()
            new_lrs = [param_group["lr"] for param_group in optimizer_edit.param_groups]
            log_string = f"Adjusted learning rate | learning rates: {new_lrs} "
            log.info(log_string)

        # generated rendered feedback visualizations
        if (
                global_step % feedback_freq == 0
                or global_step == 1
                or global_step == num_iterations
        ):
            log.info(
                f"TIME CHECK: time spent actually training "
                f"till now: {timedelta(seconds=time_spent_actually_training)}"
            )
            with torch.no_grad():
                if not feedback_pose_given:
                    render_feedback_pose = pose

                visualize_sh_vox_grid_vol_mod_rendered_feedback_attn(
                    vol_mod=vol_mod_edit,
                    vol_mod_name="attn",
                    render_feedback_pose=render_feedback_pose,
                    camera_intrinsics=camera_intrinsics,
                    global_step=global_step,
                    feedback_logs_dir=render_dir,
                    parallel_rays_chunk_size=vol_mod_edit.render_config.parallel_rays_chunk_size,
                    training_time=time_spent_actually_training,
                    log_diffuse_rendered_version=apply_diffuse_render_regularization,
                    use_optimized_sampling_mode=False,  # testing how the optimized sampling mode rendering looks 🙂
                    overridden_num_samples_per_ray=vol_mod_edit.render_config.render_num_samples_per_ray,
                    verbose_rendering=verbose_rendering,
                    log_wandb=log_wandb,
                )

            # save the model
            if (
                    global_step % save_freq == 0
                    or global_step == 1
                    or global_step == num_iterations
            ):
                log.info(
                    f"saving model-snapshot at global step {global_step}"
                )
                torch.save(
                    vol_mod_edit.get_save_info(
                        extra_info={
                            CAMERA_BOUNDS: camera_bounds,
                            CAMERA_INTRINSICS: camera_intrinsics,
                            HEMISPHERICAL_RADIUS: train_dataset.get_hemispherical_radius_estimate(),
                        }
                    ),
                    model_dir / f"model_edit_iter_{global_step}.pth",
                )
                torch.save(
                    vol_mod_object.get_save_info(
                        extra_info={
                            CAMERA_BOUNDS: camera_bounds,
                            CAMERA_INTRINSICS: camera_intrinsics,
                            HEMISPHERICAL_RADIUS: train_dataset.get_hemispherical_radius_estimate(),
                        }
                    ),
                    model_dir / f"model_object_iter_{global_step}.pth",
                )

        # ignore all the time spent doing verbose stuff :) and update
        # the last_time clock event
        last_time = time.perf_counter()

    # --------------------------------------
    #  Perform graph cut and refine grid:  |
    # --------------------------------------

        log.info(f"Starting Grid Refinement!")
        get_edit_region(vol_mod_edit=vol_mod_edit,
                        vol_mod_object=vol_mod_object,
                        vol_mod_output=vol_mod_output,
                        rays=rays_batch,
                        img_height=im_h,
                        img_width=im_w,
                        step=global_step,
                        K=kval, edit_mask_thresh=edit_mask_thresh,
                        num_obj_voxels_thresh=num_obj_voxels_thresh, min_num_edit_voxels=min_num_edit_voxels,
                        top_k_edit_thresh=top_k_edit_thresh, top_k_obj_thresh=top_k_obj_thresh)

    # change densities and features without optimization:
    regular_density = vol_mod_ref.thre3d_repr._densities.detach()
    regular_features = vol_mod_ref.thre3d_repr._features.detach()
    keep_mask = vol_mod_output.thre3d_repr.attn != 0

    new_density = vol_mod_output.thre3d_repr._densities.detach()
    new_density[keep_mask.squeeze()] = regular_density[keep_mask.squeeze()]
    vol_mod_output.thre3d_repr._densities = torch.nn.Parameter(new_density)

    new_features = vol_mod_output.thre3d_repr.features.detach()
    new_features[keep_mask.squeeze()] = regular_features[keep_mask.squeeze()]
    vol_mod_output.thre3d_repr._features = torch.nn.Parameter(new_features)

    visualize_sh_vox_grid_vol_mod_rendered_feedback(
        vol_mod=vol_mod_output,
        vol_mod_name="sds_refined",
        render_feedback_pose=render_feedback_pose,
        camera_intrinsics=camera_intrinsics,
        global_step=0,
        feedback_logs_dir=render_dir,
        parallel_rays_chunk_size=vol_mod_output.render_config.parallel_rays_chunk_size,
        training_time=time_spent_actually_training,
        log_diffuse_rendered_version=apply_diffuse_render_regularization,
        use_optimized_sampling_mode=False,  # testing how the optimized sampling mode rendering looks 🙂
        overridden_num_samples_per_ray=vol_mod_output.render_config.render_num_samples_per_ray,
        verbose_rendering=verbose_rendering,
        log_wandb=log_wandb,
    )
    vol_mod_output.thre3d_repr.attn = None

    # ------------------------
    #  Save model and exit:  |
    # ------------------------

    # save the final trained model
    log.info(f"Saving the final model-snapshot :)! Almost there ... yay!")
    torch.save(
        vol_mod_edit.get_save_info(
            extra_info={
                "camera_bounds": camera_bounds,
                "camera_intrinsics": camera_intrinsics,
                "hemispherical_radius": train_dataset.get_hemispherical_radius_estimate(),
            }
        ),
        model_dir / f"model_final_attn_edit.pth",
    )
    torch.save(
        vol_mod_object.get_save_info(
            extra_info={
                "camera_bounds": camera_bounds,
                "camera_intrinsics": camera_intrinsics,
                "hemispherical_radius": train_dataset.get_hemispherical_radius_estimate(),
            }
        ),
        model_dir / f"model_final_attn_object.pth",
    )
    torch.save(
        vol_mod_output.get_save_info(
            extra_info={
                "camera_bounds": camera_bounds,
                "camera_intrinsics": camera_intrinsics,
                "hemispherical_radius": train_dataset.get_hemispherical_radius_estimate(),
            }
        ),
        model_dir / f"model_final.pth",
    )

    # training complete yay! :)
    log.info("Training complete")
    log.info(
        f"Total actual training time: {timedelta(seconds=time_spent_actually_training)}"
    )
    return

def get_dir_batch_from_poses(poses: Tensor):
    dir_batch = []
    num_poses = poses.shape[0]
    for i in range(num_poses):
        Rt = poses[i]
        pitch, yaw = _pitch_yaw_from_Rt(Rt)

        # determine view direction according to pitch, yaw
        dir = 'front'
        if yaw > 60.0:
            dir = 'side'
        if yaw > 120.0:
            dir = 'back'
        if pitch > 55.0:
            dir = 'overhead'

        dir_batch.append(dir)

    return dir_batch


def _pitch_yaw_from_Rt(rotation: Tensor):
    # pitch = np.arccos(rotation[1, 1].cpu().numpy()) * 180.0 / np.pi
    tx, ty, tz = rotation[:, -1].cpu().numpy()
    tr = np.sqrt(tx ** 2 + ty ** 2)
    pitch = np.arctan(tz / tr) * 180 / np.pi
    yaw = np.arccos(rotation[0, 0].cpu().numpy()) * 180.0 / np.pi
    return pitch, yaw


def _tv_loss_on_grid(grid: Tensor):
    tv0 = grid.diff(dim=0).abs()
    tv1 = grid.diff(dim=1).abs()
    tv2 = grid.diff(dim=2).abs()
    return (tv0.mean() + tv1.mean() + tv2.mean()) / 3
