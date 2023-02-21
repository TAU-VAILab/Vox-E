#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train_and_render() {
	# Train:
	echo "Starting Training..."
	python initalize_3dunet_grid.py \
	-d ../data/${1}/ \
	-o logs/rf/${1}_3dunet_shdeg_0_gvg_${2}_0.25/ \
	-i logs/rf/${1}_ref_shdeg_0/saved_models/model_final.pth \
	--gvg_weight=${2}

	# render
	echo "Starting Rendering..."
	python render_3dunet_voxel_grid.py \
	-i logs/rf/${1}_3dunet_shdeg_0_gvg_${2}_0.25/saved_models/model_best.pth \
	-r logs/rf/${1}_ref_shdeg_0/saved_models/model_final.pth \
	-o output_renders/${1}_3dunet_shdeg_0_gvg_${2}_0.25
}

# STARTING RUN:
scene=dog2
gvg_weight=0.0

train_and_render $scene $gvg_weight
