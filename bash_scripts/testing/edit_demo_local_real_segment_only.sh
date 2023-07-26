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
	python refine_attn_relu_field.py \
	-d ./data/${1}/ \
	-o logs/rf/${1}/${3}/segtest/ \
	-r logs/rf/${1}/ref_real_200/saved_models/model_final.pth \
	-i /storage/etaisella/repos/Vox-E/logs/rf/${1}/${3}/saved_models/model_final.pth \
	-ie logs/rf/${1}/${3}/saved_models/model_final_attn_edit.pth \
	-io logs/rf/${1}/${3}/saved_models/model_final_attn_object.pth \
	--log_wandb=False \
    --downsample_refine_grid=True \
	--edit_mask_thresh=${4} \
    --num_obj_voxels_thresh=${5} \
	--top_k_edit_thresh=${6} \
	--top_k_obj_thresh=${7}

	# Rendering Output Video refined:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid_attn.py \
	-i logs/rf/${1}/${3}/segtest/saved_models/model_final_refined.pth \
	-o output_renders/${1}/${3}/segtest/refined/ \
	--save_freq=10
}

# STARTING RUN:

scene=pinecone
prompt="a photo of a pineapple on the ground in a backyard"
log_name="pineapple"
edit_mask_thresh=1.0
num_obj_voxels_thresh=40000
top_k_edit_thresh=300
top_k_obj_thresh=400


train_default $scene "$prompt" $log_name $edit_mask_thresh $num_obj_voxels_thresh $top_k_edit_thresh $top_k_obj_thresh

#scene=flowers
#prompt="a photo of a vase of sunflowers on the ground in a backyard"
#log_name="sunflowers2"
#eidx=7
#hf_auth_token=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
#edit_mask_thresh=0.999
#num_obj_voxels_thresh=9000
#
#train_default $scene "$prompt" $log_name $eidx $hf_auth_token $edit_mask_thresh $num_obj_voxels_thresh