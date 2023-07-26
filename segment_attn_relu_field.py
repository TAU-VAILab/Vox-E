from pathlib import Path

import click
import torch
import wandb
import copy
from datetime import datetime
from easydict import EasyDict
from torch.backends import cudnn
from torch.utils.data import DataLoader

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.modules.attn_grid_trainer import refine_edited_relu_field
from thre3d_atom.modules.volumetric_model import (
    create_volumetric_model_from_saved_model_attn,
    create_volumetric_model_from_saved_model,
)

from thre3d_atom.visualizations.static import (
    visualize_sh_vox_grid_vol_mod_rendered_feedback,
    visualize_sh_vox_grid_vol_mod_rendered_feedback_attn,
)

from thre3d_atom.data.utils import infinite_dataloader

from thre3d_atom.thre3d_reprs.voxels import VoxelGrid, VoxelSize, VoxelGridLocation, \
    create_voxel_grid_from_saved_info_dict_attn, create_voxel_grid_from_saved_info_dict
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import log_config_to_disk

from thre3d_atom.modules.refinement_functions import (
    get_edit_region,
)

from thre3d_atom.utils.imaging_utils import CameraPose

# Age-old custom option for fast training :)
cudnn.benchmark = True
# Also set torch's multiprocessing start method to spawn
# refer -> https://github.com/pytorch/pytorch/issues/40403
# for more information. Some stupid PyTorch stuff to take care of
torch.multiprocessing.set_start_method("spawn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()
# Required arguments:
@click.option("-d", "--data_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path to the input dataset")
@click.option("-ie", "--edit_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the pre-trained sds model")
@click.option("-io", "--object_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the pre-trained sds model")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for training output")
@click.option("-r", "--ref_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the pre-trained model")
@click.option("-i", "--sds_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the pre-trained sds model")

# Input dataset related arguments:
@click.option("--separate_train_test_folders", type=click.BOOL, required=False,
              default=True, help="whether the data directory has separate train and test folders",
              show_default=True)
@click.option("--data_downsample_factor", type=click.FloatRange(min=1.0), required=False,
              default=3.0, help="downscale factor for the input images if needed."
                                "Note the default, for training NeRF-based scenes", show_default=True)


# sds specific stuff
@click.option("--downsample_refine_grid", type=click.BOOL, required=False, default=False,
              help="whether to downsample the attn grid when refining (good for real scenes)",
              show_default=True)
@click.option("--kval", type=click.FLOAT, required=False, default=5.0,
              help="k value used in graphcut", show_default=True)
@click.option("--edit_mask_thresh", type=click.FLOAT, required=False, default=0.992,
              help="probability threshold for edit voxels in graph cut stage", show_default=True)
@click.option("--num_obj_voxels_thresh", type=click.INT, required=False, default=5000,
              help="number of voxels to mark as object in graph cut stage", show_default=True)
@click.option("--min_num_edit_voxels", type=click.INT, required=False, default=300,
              help="minimum number of voxels to mark as edit in graph cut stage", show_default=True)
@click.option("--top_k_edit_thresh", type=click.INT, required=False, default=300,
              help="number of voxels to mark as edit in graph cut stage if less than minimum reached", show_default=True)
@click.option("--top_k_obj_thresh", type=click.INT, required=False, default=200,
              help="number of voxels to mark as object in graph cut stage if less than minimum reached", show_default=True)

# wandb stuff
@click.option("--log_wandb", type=click.BOOL, required=False, default=False,
              help="whether to use white background for training with synthetic (background-less) scenes :)",
              show_default=True) 
@click.option("--wandb_username", type=click.STRING, required=False, default="etaisella", 
              help="wandb user name used for logging", show_default=True)
@click.option("--wandb_project_name", type=click.STRING, required=False, default="Vox-E-refine",
              help="sds prompt used for SDS based loss", show_default=True)

# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)

    # set wandb login info if required:
    if config.log_wandb:
        wandb.init(project=config.wandb_project_name, entity=config.wandb_username,
                   config=dict(config), name="test " + str(datetime.now()),
                   id=wandb.util.generate_id())

    # parse os-checked path-strings into Pathlike Paths :)
    ref_model_path = Path(config.ref_model_path)
    output_path = Path(config.output_path)
    sds_model_path = Path(config.sds_model_path)
    edit_model_path = Path(config.edit_model_path)
    object_model_path = Path(config.object_model_path)

    # save a copy of the configuration for reference
    log.info("logging configuration file ...")
    log_config_to_disk(config, output_path)

    data_path = Path(config.data_path)
    if config.separate_train_test_folders:
        train_dataset = PosedImagesDataset(
                images_dir=data_path / "train",
                camera_params_json=data_path / f"train_camera_params.json",
                normalize_scene_scale=False,
                downsample_factor=1.0,
                rgba_white_bkgd=True,
        )
    else:
        train_dataset = PosedImagesDataset(
            images_dir=data_path / "images",
            camera_params_json=data_path / "camera_params.json",
            normalize_scene_scale=False,
            downsample_factor=1.0,
            rgba_white_bkgd=True,
        )

    vol_mod_ref, _ = create_volumetric_model_from_saved_model(
        model_path=ref_model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
        device=device,
    )

    vol_mod_edit, _ = create_volumetric_model_from_saved_model_attn(
            model_path=edit_model_path,
            thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
            device=device,
            load_attn=True,
        )

    vol_mod_obj, _ = create_volumetric_model_from_saved_model_attn(
        model_path=object_model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
        device=device,
        load_attn=True,
    )

    vol_mod_output, _ = create_volumetric_model_from_saved_model_attn(
        model_path=sds_model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Set up rendering pose                                                   |
    # -------------------------------------------------------------------------

    train_dl = _make_dataloader_from_dataset(
        train_dataset, 8, 4
    )
    infinite_train_dl = iter(infinite_dataloader(train_dl))

    _, poses, _ = next(infinite_train_dl)
    selected_pose = poses[0]
    render_feedback_pose = CameraPose(rotation=selected_pose[:, :3], translation=selected_pose[:, 3:])

    # setup output directories
    # fmt: off
    model_dir = output_path / "saved_models"
    logs_dir = output_path / "training_logs"
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

    # ---------------------------------------------------------------------------
    # Find the editing region                                                   |
    # ---------------------------------------------------------------------------

    log.info(f"Starting Grid Refinement!")
    get_edit_region(vol_mod_edit=vol_mod_edit,
                    vol_mod_object=vol_mod_obj,
                    vol_mod_output=vol_mod_output,
                    K=config.kval, 
                    edit_mask_thresh=config.edit_mask_thresh,
                    num_obj_voxels_thresh=config.num_obj_voxels_thresh, 
                    min_num_edit_voxels=config.min_num_edit_voxels,
                    top_k_edit_thresh=config.top_k_edit_thresh, 
                    top_k_obj_thresh=config.top_k_obj_thresh,
                    downsample_grid=config.downsample_refine_grid)

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

    visualize_sh_vox_grid_vol_mod_rendered_feedback_attn(
                    vol_mod=vol_mod_output,
                    vol_mod_name="attn_final",
                    render_feedback_pose=render_feedback_pose,
                    camera_intrinsics=camera_intrinsics,
                    global_step=0,
                    feedback_logs_dir=render_dir,
                    parallel_rays_chunk_size=vol_mod_edit.render_config.parallel_rays_chunk_size,
                    training_time=0.0,
                    log_diffuse_rendered_version=True,
                    use_optimized_sampling_mode=False,  # testing how the optimized sampling mode rendering looks 🙂
                    overridden_num_samples_per_ray=vol_mod_edit.render_config.render_num_samples_per_ray,
                    verbose_rendering=False,
                    log_wandb=config.log_wandb,
                )

    visualize_sh_vox_grid_vol_mod_rendered_feedback(
        vol_mod=vol_mod_output,
        vol_mod_name="sds_refined",
        render_feedback_pose=render_feedback_pose,
        camera_intrinsics=camera_intrinsics,
        global_step=0,
        feedback_logs_dir=render_dir,
        parallel_rays_chunk_size=vol_mod_output.render_config.parallel_rays_chunk_size,
        training_time=0.0,
        log_diffuse_rendered_version=True,
        use_optimized_sampling_mode=False,  # testing how the optimized sampling mode rendering looks 🙂
        overridden_num_samples_per_ray=vol_mod_output.render_config.render_num_samples_per_ray,
        verbose_rendering=False,
        log_wandb=config.log_wandb,
    )

    # ------------------------
    #  Save model and exit:  |
    # ------------------------

    # save the final trained model
    log.info(f"Saving the final model-snapshot :)! Almost there ... yay!")
    torch.save(
        vol_mod_output.get_save_info(
            extra_info={
                "camera_bounds": camera_bounds,
                "camera_intrinsics": camera_intrinsics,
                "hemispherical_radius": train_dataset.get_hemispherical_radius_estimate(),
            }
        ),
        model_dir / f"model_final_refined.pth",
    )


def _make_dataloader_from_dataset(
    dataset: PosedImagesDataset, batch_size: int, num_workers: int = 0
) -> DataLoader:
    # setup the data_loader:
    # There are a bunch of fancy CPU-GPU configuration being done here.
    # Nothing too hard to understand, just refer the documentation page of PyTorch's
    # dataloader -> https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # And, read the book titled "CUDA_BY_EXAMPLE" https://developer.nvidia.com/cuda-example
    # Takes not long, just about 1-2 weeks :). But worth it :+1: :+1: :smile:!
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0 if dataset.cached_data_mode else dataset,
        pin_memory=not dataset.cached_data_mode and num_workers > 0,
        prefetch_factor=num_workers
        if not dataset.cached_data_mode and num_workers > 0
        else 2,
        persistent_workers=not dataset.cached_data_mode and num_workers > 0,
    )


if __name__ == "__main__":
    main()
