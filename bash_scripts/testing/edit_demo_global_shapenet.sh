#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g:d: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train_default() {
	# Train:
	python edit_pretrained_relu_field.py \
	-d ./data/${1}/ \
	-o logs/rf/${1}/${3}/ \
	-i logs/rf/${1}/ref/saved_models/model_final.pth \
	-p "$2" \
	--density_correlation_weight=10.0 \
	--learning_rate=0.02 \
	--log_wandb=False

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}/${3}/saved_models/model_final.pth \
	-o output_renders/${1}/${3}/
}

# STARTING RUN:

#scene=wood_chair
#prompt="a render of a wooden chair with very short legs"
#log_name="shortlegs"
#
#train_default $scene "$prompt" $log_name
#
#scene=wood_chair
#prompt="a render of a wooden chair with a round back"
#log_name="roundback"
#
#train_default $scene "$prompt" $log_name

scene=white_chair
prompt="a render of a modern white chair with a hole in it's back"
log_name="hole"

train_default $scene "$prompt" $log_name

scene=white_chair
prompt="a render of a modern white chair with a straight back"
log_name="straightback"

train_default $scene "$prompt" $log_name