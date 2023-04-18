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
	-d data/${1}/ \
	-o logs/rf/${1}/${3}/ \
	-i logs/rf/${1}/ref/saved_models/model_final.pth \
	-p "$2" \
	-eidx ${4} \
	--log_wandb=False \
	--do_refinement=True

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid_attn.py \
	-i logs/rf/${1}/${3}/saved_models/model_final_refined.pth \
	-o output_renders/${1}/${3}/
}

# STARTING RUN:

scene=dog2
prompt="a render of a dog with a party hat"
log_name="party_hat"
eidx=9
oidx=5

train_default $scene "$prompt" $log_name $eidx