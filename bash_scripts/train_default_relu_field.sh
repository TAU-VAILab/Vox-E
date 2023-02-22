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
	python train_sh_based_voxel_grid_with_posed_images.py -d ../data/${1}/ \
	-o logs/rf/${1}_ref_shdeg_${2}/ \
	--sh_degree=0 \
	--lpips_weight=${3}

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}_ref_shdeg_${2}/saved_models/model_final.pth \
	-o output_renders/${1}_ref_shdeg_${2}
}

# STARTING RUN:

scene=$scene_in
sh_degree=0
lpips_weight=0.0

train_default $scene $sh_degree $lpips_weight