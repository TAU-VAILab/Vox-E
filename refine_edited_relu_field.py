from pathlib import Path

import click
import torch
import wandb
import copy
from datetime import datetime
from easydict import EasyDict
from torch.backends import cudnn

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.modules.attn_grid_trainer import refine_edited_relu_field
from thre3d_atom.modules.volumetric_model import (
    create_volumetric_model_from_saved_model_attn,
    create_volumetric_model_from_saved_model,
)


from thre3d_atom.thre3d_reprs.voxels import VoxelGrid, VoxelSize, VoxelGridLocation, \
    create_voxel_grid_from_saved_info_dict_attn, create_voxel_grid_from_saved_info_dict
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import log_config_to_disk

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
@click.option("-i", "--sds_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the pre-trained sds model")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for training output")
@click.option("-r", "--ref_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the pre-trained model")
@click.option("-a", "--hf_auth_token", type=click.STRING, required=False, default="",
              help="hugging face model token for stable diffusion 1.4",
              show_default=True)
@click.option("-p", "--prompt", type=click.STRING, required=True,
              help="prompt used for attention")
@click.option("-eidx", "--edit_idx", required=True, type=click.STRING, 
              help="index of edit item, i.e. hat")
@click.option("-oidx", "--object_idx", type=click.INT, required=False, default=None,
              help="index of object, i.e. cat")
@click.option("-t", "--timestamp", type=click.INT, required=False, default=200,
              help="diffusion_timestamp")
# Input dataset related arguments:
@click.option("--separate_train_test_folders", type=click.BOOL, required=False,
              default=True, help="whether the data directory has separate train and test folders",
              show_default=True)
@click.option("--data_downsample_factor", type=click.FloatRange(min=1.0), required=False,
              default=3.0, help="downscale factor for the input images if needed."
                                "Note the default, for training NeRF-based scenes", show_default=True)
# Voxel-grid related arguments:
@click.option("--grid_dims", type=click.INT, nargs=3, required=False, default=(160, 160, 160),
              help="dimensions (#voxels) of the grid along x, y and z axes", show_default=True)
@click.option("--grid_location", type=click.FLOAT, nargs=3, required=False, default=(0.0, 0.0, 0.0),
              help="dimensions (#voxels) of the grid along x, y and z axes", show_default=True)
@click.option("--normalize_scene_scale", type=click.BOOL, required=False, default=False,
              help="whether to normalize the scene's scale to unit radius", show_default=True)
@click.option("--grid_world_size", type=click.FLOAT, nargs=3, required=False, default=(3.0, 3.0, 3.0),
              help="size (extent) of the grid in world coordinate system."
                   "Please carefully note it's use in conjunction with the normalization :)", show_default=True)
@click.option("--sh_degree", type=click.INT, required=False, default=0,
              help="degree of the spherical harmonics coefficients to be used. "
                   "Supported values: [0, 1, 2, 3]", show_default=True)
# -------------------------------------------------------------------------------------
#                        !!! :) MOST IMPORTANT OPTION :) !!!                          |
# -------------------------------------------------------------------------------------
@click.option("--use_relu_field", type=click.BOOL, required=False, default=True,  # |
              help="whether to use relu_fields or revert to traditional grids",  # |
              show_default=True)  # |
# -------------------------------------------------------------------------------------

@click.option("--use_softplus_field", type=click.BOOL, required=False, default=True,
              help="whether to use softplus_field or relu_field", show_default=True)
# Rendering related arguments:
@click.option("--render_num_samples_per_ray", type=click.INT, required=False, default=1024,
              help="number of samples taken per ray during rendering", show_default=True)
@click.option("--parallel_rays_chunk_size", type=click.INT, required=False, default=32768,
              help="number of parallel rays processed on the GPU for accelerated rendering", show_default=True)
@click.option("--white_bkgd", type=click.BOOL, required=False, default=True,
              help="whether to use white background for training with synthetic (background-less) scenes :)",
              show_default=True)  # this option is also used in pre-processing the dataset

# Training related arguments:
@click.option("--ray_batch_size", type=click.INT, required=False, default=84672,
              help="number of randomly sampled rays used per training iteration", show_default=True)
@click.option("--train_num_samples_per_ray", type=click.INT, required=False, default=256,
              help="number of samples taken per ray during training", show_default=True)
@click.option("--num_stages", type=click.INT, required=False, default=1,
              help="number of progressive growing stages used in training", show_default=True)
@click.option("--num_iterations_per_stage", type=click.INT, required=False, default=1500,
              help="number of training iterations performed per stage", show_default=True)
@click.option("--scale_factor", type=click.FLOAT, required=False, default=2.0,
              help="factor by which the grid is up-scaled after each stage", show_default=True)
@click.option("--learning_rate", type=click.FLOAT, required=False, default=0.028,
              help="learning rate used at the beginning (ADAM OPTIMIZER)", show_default=True)
@click.option("--lr_decay_steps_per_stage", type=click.INT, required=False, default=5000 * 100,
              help="number of iterations after which lr is exponentially decayed per stage", show_default=True)
@click.option("--lr_decay_gamma_per_stage", type=click.FLOAT, required=False, default=0.1,
              help="value of gamma for exponential lr_decay (happens per stage)", show_default=True)
@click.option("--stagewise_lr_decay_gamma", type=click.FLOAT, required=False, default=0.9,
              help="value of gamma used for reducing the learning rate after each stage", show_default=True)
@click.option("--apply_diffuse_render_regularization", type=click.BOOL, required=False, default=True,
              help="whether to apply the diffuse render regularization."
                   "this is a weird conjure of mine, where we ask the diffuse render "
                   "to match, as closely as possible, the GT-possibly-specular one :D"
                   "can be off or on, on yields stabler training :) ", show_default=False)
@click.option("--num_workers", type=click.INT, required=False, default=4,
              help="number of worker processes used for loading the data using the dataloader"
                   "note that this will be ignored if GPU-caching of the data is successful :)", show_default=True)
# Various frequencies:
@click.option("--save_frequency", type=click.INT, required=False, default=250,
              help="number of iterations after which a model is saved", show_default=True)
@click.option("--test_frequency", type=click.INT, required=False, default=250,
              help="number of iterations after which test metrics are computed", show_default=True)
@click.option("--feedback_frequency", type=click.INT, required=False, default=200,
              help="number of iterations after which rendered feedback is generated", show_default=True)
@click.option("--summary_frequency", type=click.INT, required=False, default=50,
              help="number of iterations after which training-loss/other-summaries are logged", show_default=True)
# Miscellaneous modes
@click.option("--verbose_rendering", type=click.BOOL, required=False, default=False,
              help="whether to show progress while rendering feedback during training"
                   "can be turned-off when running on server-farms :D", show_default=True)
@click.option("--data_pose_mode", type=click.BOOL, required=False, default=False,
              help="uses poses from a given dataset instead of random sampling",
               show_default=True)

# sds specific stuff
@click.option("--directional_dataset", type=click.BOOL, required=False, default=True,
              help="whether to use a directional dataset for SDS where each view comes with a direction",
              show_default=True)
@click.option("--downsample_refine_grid", type=click.BOOL, required=False, default=False,
              help="whether to downsample the attn grid when refining (good for real scenes)",
              show_default=True)
@click.option("--attn_tv_weight", type=click.FLOAT, required=False, default=0.01,
              help="value of gamma for exponential lr_decay (happens per stage)", show_default=True)
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
    sds_model_path = Path(config.sds_model_path)
    ref_model_path = Path(config.ref_model_path)
    output_path = Path(config.output_path)

    # save a copy of the configuration for reference
    log.info("logging configuration file ...")
    log_config_to_disk(config, output_path)

    data_path = Path(config.data_path)
    if config.separate_train_test_folders:
        train_dataset = PosedImagesDataset(
                images_dir=data_path / "train",
                camera_params_json=data_path / f"train_camera_params.json",
                normalize_scene_scale=config.normalize_scene_scale,
                downsample_factor=config.data_downsample_factor,
                rgba_white_bkgd=config.white_bkgd,
        )
    else:
        train_dataset = PosedImagesDataset(
            images_dir=data_path / "images",
            camera_params_json=data_path / "camera_params.json",
            normalize_scene_scale=config.normalize_scene_scale,
            downsample_factor=config.data_downsample_factor,
            rgba_white_bkgd=config.white_bkgd,
        )
    
    # set up image dims
    im_h = train_dataset._camera_intrinsics.height
    im_w = train_dataset._camera_intrinsics.width
    image_dims = (im_h, im_w)

    pretrained_vol_mod, _ = create_volumetric_model_from_saved_model(
        model_path=ref_model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
        device=device,
    )

    vol_mod_edit, _ = create_volumetric_model_from_saved_model_attn(
            model_path=sds_model_path,
            thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
            device=device,
        )

    vol_mod_obj, _ = create_volumetric_model_from_saved_model_attn(
        model_path=sds_model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
        device=device,
    )

    sds_vol_mod, _ = create_volumetric_model_from_saved_model_attn(
        model_path=sds_model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
        device=device,
    )

    # convert space separated string to list of ints
    edit_idx = [int(i) for i in config.edit_idx.split()]

    # train the model:
    refine_edited_relu_field(
            vol_mod_edit=vol_mod_edit,
            vol_mod_object=vol_mod_obj,
            vol_mod_ref=pretrained_vol_mod,
            vol_mod_output=sds_vol_mod,
            train_dataset=train_dataset,
            hf_auth_token=config.hf_auth_token,
            output_dir=output_path,
            prompt=config.prompt,
            edit_idx=edit_idx,
            object_idx=config.object_idx,
            timestamp=config.timestamp,
            image_dims=image_dims,
            ray_batch_size=config.ray_batch_size,
            num_iterations=config.num_iterations_per_stage,
            learning_rate=config.learning_rate,
            save_freq=config.save_frequency,
            feedback_freq=config.feedback_frequency,
            summary_freq=config.summary_frequency,
            apply_diffuse_render_regularization=config.apply_diffuse_render_regularization,
            verbose_rendering=config.verbose_rendering,
            attn_tv_weight=config.attn_tv_weight,
            kval=config.kval,
            edit_mask_thresh=config.edit_mask_thresh,
            num_obj_voxels_thresh=config.num_obj_voxels_thresh,
            min_num_edit_voxels=config.min_num_edit_voxels,
            top_k_edit_thresh=config.top_k_edit_thresh,
            top_k_obj_thresh=config.top_k_obj_thresh,
            log_wandb=config.log_wandb,
            data_pose_mode=config.data_pose_mode,
            downsample_refine_grid=config.downsample_refine_grid,
        )


if __name__ == "__main__":
    main()
