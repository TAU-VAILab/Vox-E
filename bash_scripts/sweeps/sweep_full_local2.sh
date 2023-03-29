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
	python edit_pretrained_relu_field.py \
	-d ../data/${1}/ \
	-o ./logs/rf/${2}/${1}/${4} \
	-i ./logs/rf/high_res_${1}/saved_models/model_final.pth \
	-p "$3" \
	-eidx 6

#	# Rendering Output Video:
#	echo "Starting Rendering..."
#	python render_sh_based_voxel_grid.py \
#	-i ./logs/rf/${2}/${1}/${4}/saved_models/model_final.pth \
#	-o ./output_renders/${2}/${1}/${4}/ \
#	--sds_prompt="$3" \
#	--save_freq=10
}

# STARTING RUN:


sweep_name=hat_sweep
scene=dog2
prompt="a render of a light gray dog with a birthday hat"
log_name="birthday_hat_dog_3"

train_default $scene $sweep_name "$prompt" $log_name