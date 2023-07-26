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
	python edit_pretrained_relu_field.py \
	-d ./data/${1}/ \
	-o logs/rf/${1}/${3}/ \
	-i logs/rf/${1}/ref_real_200/saved_models/model_final.pth \
	-p "$2" \
	-eidx "$4" \
	--log_wandb=False \
	--do_refinement=True \
	--lr_freq=300 \
	--lr_decay_start=2500 \
	--lr_gamma=0.85 \
	--num_iterations_edit=4000 \
	--sds_t_start=2000 \
	--learning_rate=0.005 \
	--density_correlation_weight=60000.0 \
	--data_pose_mode=True \
    --downsample_refine_grid=True \
	--edit_mask_thresh=1.0 \
	--num_obj_voxels_thresh=40000 \
	--top_k_edit_thresh=290 \
	--top_k_obj_thresh=2500 \
	--hf_auth_token=${5}

	# Rendering Output Video refined:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid_attn.py \
	-i logs/rf/${1}/${3}/saved_models/model_final_refined.pth \
	-o output_renders/${1}/${3}/ \
	--save_freq=10
}

# STARTING RUN:

scene=pinecone
prompt="a photo of a pineapple on the ground in a backyard"
log_name="pineapple"
eidx="5 6"
hf_auth_token=$hf_auth_token_in

train_default $scene "$prompt" $log_name $eidx $hf_auth_token