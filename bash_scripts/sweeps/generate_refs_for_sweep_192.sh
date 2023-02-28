#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g:d: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
		d) sweep_name_in=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train_default() {
	# Train:
	echo "Starting Training..."
	python train_sh_based_voxel_grid_with_posed_images.py -d ../data/${1}/ \
	-o logs/rf/${2}/${1}/ref \
	--sh_degree=0 \
	--grid_dims=192 192 192 \
	--data_downsample_factor=1.0

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${2}/${1}/ref/saved_models/model_final.pth \
	-o output_renders/${2}/${1}/ref \
	--sds_prompt="$3" \
	--save_freq=10
}

# STARTING RUN:

sweep_name=sweep_192
scene=duck
prompt="a render of a duck"

train_default $scene $sweep_name "$prompt"