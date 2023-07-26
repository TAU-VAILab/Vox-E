from pathlib import Path
import click
import imageio
import torch
import os

from thre3d_atom.data.datasets import PosedImagesDataset
from torch.utils.data import DataLoader
from thre3d_atom.data.utils import infinite_dataloader
from thre3d_atom.utils.imaging_utils import CameraPose
from thre3d_atom.modules.volumetric_model import (
    create_volumetric_model_from_saved_model,
)
from thre3d_atom.thre3d_reprs.voxels import create_voxel_grid_from_saved_info_dict
from thre3d_atom.utils.constants import HEMISPHERICAL_RADIUS, CAMERA_INTRINSICS
from thre3d_atom.utils.imaging_utils import (
    get_thre360_animation_poses,
    get_thre360_spiral_animation_poses,
)
from thre3d_atom.visualizations.animations import (
    render_camera_path_for_volumetric_model,
)
from easydict import EasyDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()
# Required arguments:
@click.option("-i", "--model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the trained (reconstructed) model")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for saving rendered output")
@click.option("-r", "--ref_path", type=click.Path(file_okay=True, dir_okay=False), default=None,
              required=False, help="path for saving rendered output")

# Non-required Render configuration options:
@click.option("-d", "--data_path", type=click.Path(file_okay=False, dir_okay=True),
              required=False, help="path to the input dataset")
@click.option("--overridden_num_samples_per_ray", type=click.IntRange(min=1), default=512,
              required=False, help="overridden (increased) num_samples_per_ray for beautiful renders :)")
@click.option("--render_scale_factor", type=click.FLOAT, default=2.0,
              required=False, help="overridden (increased) resolution (again :D) for beautiful renders :)")
@click.option("--camera_path", type=click.Choice(["thre360", "spiral", "dataset"]), default="thre360",
              required=False, help="which camera path to use for rendering the animation")
# thre360_path options
@click.option("--camera_pitch", type=click.FLOAT, default=60.0,
              required=False, help="pitch-angle value for the camera for 360 path animation")
@click.option("--num_frames", type=click.IntRange(min=1), default=180,
              required=False, help="number of frames in the video")
# spiral path options
@click.option("--vertical_camera_height", type=click.FLOAT, default=3.0,
              required=False, help="height at which the camera spiralling will happen")
@click.option("--num_spiral_rounds", type=click.IntRange(min=1), default=2,
              required=False, help="number of rounds made while transitioning between spiral radii")

# Non-required video options:
@click.option("--fps", type=click.IntRange(min=1), default=60,
              required=False, help="frames per second of the video")

# Output saving additions:
@click.option("--save_freq", type=click.INT, default=None,
              required=False, help="frames per second of the video")
@click.option("-p", "--sds_prompt", type=click.STRING, required=False, default=None,
              help="sds prompt used for SDS based loss, if not None, prints it out to a file")

# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)

    # parse os-checked path-strings into Pathlike Paths :)
    model_path = Path(config.model_path)
    output_path = Path(config.output_path)

    # create the output path if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)

    # save prompt to text file if not None
    if config.sds_prompt != None:
        text_path = output_path / "prompt.txt"
        with open(text_path, 'w') as file:
            file.write(config.sds_prompt)

    # load volumetric_model from the model_path
    vol_mod, extra_info = create_volumetric_model_from_saved_model(
        model_path=model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
        device=device,
    )
    vol_mod.render_config.random_bkgd = False
    vol_mod.render_config.white_bkgd = True

    # override extra info with ref's if given - raises quality
    if config.ref_path != None:
        ref_path = Path(config.ref_path)
        _, extra_info_ref = create_volumetric_model_from_saved_model(
            model_path=ref_path,
            thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
            device=device,
        )
        extra_info = extra_info_ref

    hemispherical_radius = extra_info[HEMISPHERICAL_RADIUS]
    camera_intrinsics = extra_info[CAMERA_INTRINSICS]

    # generate animation using the newly_created vol_mod :)
    if config.camera_path == "thre360":
        camera_pitch, num_frames = config.camera_pitch, config.num_frames
        animation_poses = get_thre360_animation_poses(
            hemispherical_radius=hemispherical_radius,
            camera_pitch=camera_pitch,
            num_poses=num_frames,
        )
    elif config.camera_path == "spiral":
        vertical_camera_height, num_frames = (
            config.vertical_camera_height,
            config.num_frames,
        )
        animation_poses = get_thre360_spiral_animation_poses(
            horizontal_radius_range=(hemispherical_radius / 8.0, hemispherical_radius),
            vertical_camera_height=vertical_camera_height,
            num_rounds=config.num_spiral_rounds,
            num_poses=num_frames,
        )
    elif config.camera_path == "dataset":
        print("using dataset poses!")
        data_path = Path(config.data_path)
        image_path = data_path / "train"
        train_dataset = PosedImagesDataset(
                images_dir=image_path,
                camera_params_json=data_path / f"train_camera_params.json",
                normalize_scene_scale=False,
                downsample_factor=1.0,
                rgba_white_bkgd=vol_mod.render_config.white_bkgd,
        )
        num_frames = len(os.listdir(image_path))
        train_dl = _make_dataloader_from_dataset(
            train_dataset, num_frames, 4
        )
        infinite_train_dl = iter(infinite_dataloader(train_dl))
        _, poses, _ = next(infinite_train_dl)
        animation_poses = [CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]) for pose in poses]
    else:
        raise ValueError(
            f"Unknown camera_path ``{config.camera_path}'' requested."
            f"Only available options are: ['thre360' and 'spiral']"
        )

    animation_frames = render_camera_path_for_volumetric_model(
        vol_mod=vol_mod,
        camera_path=animation_poses,
        camera_intrinsics=camera_intrinsics,
        overridden_num_samples_per_ray=config.overridden_num_samples_per_ray,
        render_scale_factor=config.render_scale_factor,
        image_save_freq=config.save_freq,
        image_save_path=output_path,
    )

    imageio.mimwrite(
        output_path / "rendered_video.mp4",
        animation_frames,
        fps=config.fps,
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
