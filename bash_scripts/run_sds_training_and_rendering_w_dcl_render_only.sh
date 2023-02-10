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
	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}_sds_dir_${3}_dcl_${5}_${4}_front_overhead/saved_models/model_stage_1_iter_2000.pth \
	-o output_renders/${1}_sds_dir_${3}_dcl_${5}_${4}_front_overhead
}

# STARTING RUN:

scene=dog2
prompt="a render of a mean looking light greydog"
directional=True
log_name="mean" # 1-word description of the prompt for saving
dcl_weight=1500.0

train_and_render $scene "$prompt" $directional $log_name $dcl_weight
