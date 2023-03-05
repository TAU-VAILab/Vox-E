from pathlib import Path

import click
import torch
import wandb
import copy
from datetime import datetime
from easydict import EasyDict
from torch.backends import cudnn

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.modules.sds_trainer import train_sh_vox_grid_vol_mod_with_posed_images_and_sds
from thre3d_atom.modules.attn_grid_trainer import refine_edited_relu_field
from thre3d_atom.modules.volumetric_model import (
    create_volumetric_model_from_saved_model,
    create_volumetric_model_from_saved_model_attn,
)

from thre3d_atom.thre3d_reprs.voxels import (
    create_voxel_grid_from_saved_info_dict,
    create_voxel_grid_from_saved_info_dict_attn,
)
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
@click.option("-i", "--pretrained_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the pre-trained relu field model")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for training output")
@click.option("-p", "--prompt", type=click.STRING, required=True,
              help="sds prompt used for SDS based loss")

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
#                          !!! :)  IMPORTANT OPTION :) !!!                            |
# -------------------------------------------------------------------------------------
@click.option("--use_relu_field", type=click.BOOL, required=False, default=True,    # |
              help="whether to use relu_fields or revert to traditional grids",     # |
              show_default=True)                                                    # |              
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
@click.option("--ray_batch_size", type=click.INT, required=False, default=65536 * 1.25,
              help="number of randomly sampled rays used per training iteration", show_default=True)
@click.option("--train_num_samples_per_ray", type=click.INT, required=False, default=256,
              help="number of samples taken per ray during training", show_default=True)
@click.option("--num_stages", type=click.INT, required=False, default=1,
              help="number of progressive growing stages used in training", show_default=True)
@click.option("--num_iterations_edit", type=click.INT, required=False, default=8000,
              help="number of training iterations performed in the editing (SDS) stage", show_default=True)
@click.option("--scale_factor", type=click.FLOAT, required=False, default=2.0,
              help="factor by which the grid is up-scaled after each stage", show_default=True)
@click.option("--learning_rate", type=click.FLOAT, required=False, default=0.025,
              help="learning rate used at the beginning (ADAM OPTIMIZER)", show_default=True)
@click.option("--lr_freq", type=click.INT, required=False, default=400,
              help="frequency in which to reduce learning rate", show_default=True)
@click.option("--lr_decay_start", type=click.INT, required=False, default=5000,
              help="step in which to start decreasing learning rate", show_default=True)
@click.option("--lr_gamma", type=click.FLOAT, required=False, default=0.96,
              help="value of gamma for exponential lr_decay (happens per stage)", show_default=True)
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
@click.option("--feedback_frequency", type=click.INT, required=False, default=100,
              help="number of iterations after which rendered feedback is generated", show_default=True)
@click.option("--summary_frequency", type=click.INT, required=False, default=50,
              help="number of iterations after which training-loss/other-summaries are logged", show_default=True)

# Miscellaneous modes
@click.option("--verbose_rendering", type=click.BOOL, required=False, default=False,
              help="whether to show progress while rendering feedback during training"
                   "can be turned-off when running on server-farms :D", show_default=True)
@click.option("--fast_debug_mode", type=click.BOOL, required=False, default=False,
              help="whether to use the fast debug mode while training "
                   "(skips testing and some lengthy visualizations)", show_default=True)

# sds specific stuff
@click.option("--diffuse_weight", type=click.FLOAT, required=False, default=0.0000001,
              help="diffuse weight used for regularization", show_default=True)
@click.option("--specular_weight", type=click.FLOAT, required=False, default=0.0000001,
              help="specular weight used for regularization", show_default=True)
@click.option("--directional_dataset", type=click.BOOL, required=False, default=True,
              help="whether to use a directional dataset for SDS where each view comes with a direction",
               show_default=True)
@click.option("--use_uncertainty", type=click.BOOL, required=False, default=False,
              help="whether to use an uncertainty aware type loss",
               show_default=True)
@click.option("--do_sds", type=click.BOOL, required=False, default=True,
              help="whether to use an uncertainty aware type loss",
               show_default=True)
@click.option("--new_frame_frequency", type=click.INT, required=False, default=1,
              help="number of iterations where we work on the same pose", show_default=True)
@click.option("--density_correlation_weight", type=click.FLOAT, required=False, default=200.0,
              help="weight for density correlation loss", show_default=True)
@click.option("--feature_correlation_weight", type=click.FLOAT, required=False, default=0.0,
              help="weight for feature correlation loss", show_default=True)
@click.option("--tv_density_weight", type=click.FLOAT, required=False, default=0.0,
              help="weight for total variation loss on densities", show_default=True)
@click.option("--tv_features_weight", type=click.FLOAT, required=False, default=0.0,
              help="weight for total variation loss on densities", show_default=True)

# sds timestep scheduling:
@click.option("--sds_t_freq", type=click.INT, required=False, default=600,
              help="frequency in which to reduce the max timestep in sds", show_default=True)
@click.option("--sds_t_start", type=click.INT, required=False, default=4000,
              help="iteration in which to start reducing the max timestep in sds", show_default=True)
@click.option("--sds_t_gamma", type=click.FLOAT, required=False, default=0.75,
              help="max timestep reduction gamma", show_default=True)

# refinement:
# -------------------------------------------------------------------------------------
#                        !!! :) MOST IMPORTANT OPTION :) !!!                          |
# -------------------------------------------------------------------------------------
@click.option("--do_refinement", type=click.BOOL, required=False, default=False,    # |
              help="whether to use the refinement stage for improving local edits", # |
              show_default=True)                                                    # |              
# -------------------------------------------------------------------------------------

@click.option("--kval", type=click.FLOAT, required=False, default=5.0,
              help="k value used in graphcut", show_default=True)
@click.option("--num_iterations_refine", type=click.INT, required=False, default=1500,
              help="number of training iterations performed in the refinement stage", show_default=True)

# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)

    wandb.init(project='VoxelArtReluFields v1.1', entity="etaisella",
               config=dict(config), name="test " + str(datetime.now()),
               id=wandb.util.generate_id())
    # parse os-checked path-strings into Pathlike Paths :)
    data_path = Path(config.data_path)
    model_path = Path(config.high_res_model_path)
    output_path = Path(config.output_path)

    # save a copy of the configuration for reference
    log.info("logging configuration file ...")
    log_config_to_disk(config, output_path)

    if config.separate_train_test_folders:
        train_dataset, test_dataset = (
            PosedImagesDataset(
                images_dir=data_path / mode,
                camera_params_json=data_path / f"{mode}_camera_params.json",
                normalize_scene_scale=config.normalize_scene_scale,
                downsample_factor=config.data_downsample_factor,
                rgba_white_bkgd=config.white_bkgd,
            )
            for mode in ("train", "test")
        )
    else:
        train_dataset = PosedImagesDataset(
            images_dir=data_path / "images",
            camera_params_json=data_path / "camera_params.json",
            normalize_scene_scale=config.normalize_scene_scale,
            downsample_factor=config.data_downsample_factor,
            rgba_white_bkgd=config.white_bkgd,
        )
        test_dataset = None

    pretrained_vol_mod, _ = create_volumetric_model_from_saved_model(
        model_path=model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
        device=device,
    )

    sds_vol_mod = copy.deepcopy(pretrained_vol_mod)

    # train the model:
    train_sh_vox_grid_vol_mod_with_posed_images_and_sds(
        sds_vol_mod=sds_vol_mod,
        pretrained_vol_mod=pretrained_vol_mod,
        train_dataset=train_dataset,
        output_dir=output_path,
        test_dataset=test_dataset,
        ray_batch_size=config.ray_batch_size,
        num_stages=config.num_stages,
        num_iterations_per_stage=config.num_iterations_edit,
        scale_factor=config.scale_factor,
        learning_rate=config.learning_rate,
        lr_decay_start=config.lr_decay_start,
        lr_freq=config.lr_freq,
        lr_gamma=config.lr_gamma,
        save_freq=config.save_frequency,
        test_freq=config.test_frequency,
        feedback_freq=config.feedback_frequency,
        summary_freq=config.summary_frequency,
        apply_diffuse_render_regularization=config.apply_diffuse_render_regularization,
        num_workers=config.num_workers,
        verbose_rendering=config.verbose_rendering,
        fast_debug_mode=config.fast_debug_mode,
        diffuse_weight=config.diffuse_weight,
        specular_weight=config.specular_weight,
        sds_prompt=config.prompt,
        directional_dataset=config.directional_dataset,
        use_uncertainty=config.use_uncertainty,
        new_frame_frequency=config.new_frame_frequency,
        density_correlation_weight=config.density_correlation_weight,
        feature_correlation_weight=config.feature_correlation_weight,
        tv_density_weight=config.tv_density_weight,
        tv_features_weight=config.tv_features_weight,
        do_sds=config.do_sds,
        sds_t_freq=config.sds_t_freq,
        sds_t_start=config.sds_t_start,
        sds_t_gamma=config.sds_t_gamma,
    )

    # exit here if not going to refine stage
    if not config.do_refinement:
        return
    
    vol_mod_edit, _ = create_volumetric_model_from_saved_model_attn(
        model_path=output_path / f"saved_models" / f"model_final.pth",
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
        device=device,
    )

    vol_mod_obj, _ = create_volumetric_model_from_saved_model_attn(
        model_path=output_path / f"saved_models" / f"model_final.pth",
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
        device=device,
    )

    refine_edited_relu_field(
        vol_mod_edit=vol_mod_edit,
        vol_mod_object=vol_mod_obj,
        vol_mod_ref=pretrained_vol_mod,
        vol_mod_output=sds_vol_mod,
        train_dataset=train_dataset,
        output_dir=output_path,
        prompt=config.prompt,
        edit_idx=config.edit_idx,
        object_idx=config.object_idx,
        timestamp=config.timestamp,
        ray_batch_size=config.ray_batch_size,
        num_stages=config.num_stages,
        num_iterations_per_stage=config.num_iterations_refine,
        scale_factor=config.scale_factor,
        learning_rate=config.learning_rate,
        lr_decay_gamma_per_stage=config.lr_decay_gamma_per_stage,
        lr_decay_steps_per_stage=config.lr_decay_steps_per_stage,
        stagewise_lr_decay_gamma=config.stagewise_lr_decay_gamma,
        save_freq=config.save_frequency,
        feedback_freq=config.feedback_frequency,
        summary_freq=config.summary_frequency,
        apply_diffuse_render_regularization=config.apply_diffuse_render_regularization,
        num_workers=config.num_workers,
        verbose_rendering=config.verbose_rendering,
        directional_dataset=config.directional_dataset,
        attn_tv_weight=config.attn_tv_weight,
        kval=config.kval,
    )

if __name__ == "__main__":
    main()
