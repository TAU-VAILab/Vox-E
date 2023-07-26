#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g:d: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
		d) scene_in=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train_default() {
	# Train:
	echo "Starting Training..."
	python train_sh_based_voxel_grid_with_posed_images.py -d ./nerf_360/${1}/ \
	-o logs/rf/${1}/ref_real_200/ \
	--fast_debug_mode=True \
	--grid_dims=200 200 200 \
	--train_num_samples_per_ray=416 \
	--linear_disparity_sampling=True \
	--sh_degree=0

	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}/ref_real_200/saved_models/model_final.pth \
	-o output_renders/${1}_lds/ref_real_200/ \
	-p "$2" \
	--save_freq=10 
}

# STARTING RUN:

scene=$scene_in
sh_degree=0
lpips_weight=0.0

train_default $scene $sh_degree $lpips_weight