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
	# # Train:
	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid_attn.py \
	-d ./data/${1}/ \
	-i /storage/etaisella/repos/Vox-E/logs/rf/${1}/pineapple5/segtest/saved_models/model_final_refined.pth \
	-o output_renders/${1}/pineapple5_segtest/ \
	--camera_path="dataset" \
	--save_freq=1 \
	--ref_path=logs/rf/${1}/ref_real_200/saved_models/model_final.pth
}

# STARTING RUN:

scene=pinecone
prompt="a picture of a pinecone"
log_name="ref_real_200"
eidx=9
oidx=5
hf_auth_token=$hf_auth_token_in

train_default $scene "$prompt" $log_name $eidx $hf_auth_token


#scene=flowers
#prompt="a render of a vase of flowers"
#log_name="ref_real_200"
#eidx=9
#oidx=5
#hf_auth_token=$hf_auth_token_in
#
#train_default $scene "$prompt" $log_name $eidx $hf_auth_token
