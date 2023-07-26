#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g:a: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
    a) hf_auth_token_in=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train_default() {
	# Train:
	python refine_edited_relu_field_th.py \
	-d ./data/${1}/ \
	-o logs/rf/${1}/${3}/ \
	-r logs/rf/${1}/ref/saved_models/model_final.pth \
	-i logs/rf/${1}/${3}/saved_models/model_final.pth \
	-ie logs/rf/${1}/${3}/saved_models/model_final_attn_edit.pth \
	-io logs/rf/${1}/${3}/saved_models/model_final_attn_object.pth \
	--threshold=${4} \
	--log_wandb=False 

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid_attn.py \
	-i logs/rf/${1}/${3}/saved_models/model_final_refined_th_${4}.pth \
	-o output_renders/${1}/${3}_refined_th_${4}/ \
	--save_freq=10 

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid_attn2.py \
	-i logs/rf/${1}/${3}/saved_models/model_final_refined_th_${4}.pth \
	-ie logs/rf/${1}/${3}/saved_models/model_final_attn_edit.pth \
	-io logs/rf/${1}/${3}/saved_models/model_final_attn_object.pth \
	-p "$2" \
	--edit_idx=9 \
	--save_freq=10 \
	--hf_auth_token=${5} \
	-o output_renders/${1}/${3}_attn_wgt_th_${4}/
}

# STARTING RUN:

scene=dog2
prompt="a render of a dog with a party hat"
log_name="party_hat2"
eidx=9
hf_auth_token=$hf_auth_token_in
thresh=0.8

train_default $scene "$prompt" $log_name $thresh $eidx $hf_auth_token

scene=dog2
prompt="a render of a dog with a party hat"
log_name="party_hat2"
eidx=9
hf_auth_token=$hf_auth_token_in
thresh=0.97

train_default $scene "$prompt" $log_name $thresh $eidx $hf_auth_token

scene=dog2
prompt="a render of a dog with a party hat"
log_name="party_hat2"
eidx=9
hf_auth_token=$hf_auth_token_in
thresh=0.9

train_default $scene "$prompt" $log_name $thresh $eidx $hf_auth_token