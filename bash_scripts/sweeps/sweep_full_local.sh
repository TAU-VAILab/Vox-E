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
	-d ./data/${1}/ \
	-o logs/rf/${2}/${1}/${4} \
	-i logs/rf/${1}/ref/saved_models/model_final.pth \
	-p "$3" \
	-a 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 \
	-eidx=${5} \
	--do_refinement=True \
	--log_wandb=True \
	--learning_rate=0.028 \
	--sh_degree=0 # we currently only support diffuse

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${2}/${1}/${4}/saved_models/model_final_refined.pth \
	-o output_renders/${2}/${1}/${4}/ \
	--sds_prompt="$3" \
	--save_freq=10
}

# STARTING RUN:

sweep_name=sweep_full_local_extended
scene=duck
prompt="a render of a duck wearing a party hat"
log_name="party_hat"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_extended
scene=duck
prompt="a render of a duck wearing a christmas sweater"
log_name="christmas"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_extended
scene=duck
prompt="a render of a duck wearing big sunglasses"
log_name="sunglasses"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx
