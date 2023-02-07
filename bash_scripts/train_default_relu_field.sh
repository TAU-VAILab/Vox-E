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
	-o logs/rf/high_res_${1}_diffuse/ \
	--sh_degree=0 # we currently only support diffuse
}

# STARTING RUN:

scene=$scene_in

train_default $scene