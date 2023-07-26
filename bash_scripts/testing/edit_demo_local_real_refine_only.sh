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
	python refine_attn_relu_field.py \
	--data_path=./data/${1}/ \
    --output_path=./logs/rf/${1}/${2}/ \
    --sds_model_path=./logs/rf/${1}/${2}/model_final.pth \
    --edit_model_path=./logs/rf/${1}/${2}/model_final_attn_edit.pth \
    --object_model_path=./logs/rf/${1}/${2}/model_final_attn_object.pth \
    --ref_model_path=./logs/rf/${1}/ref_real_200/saved_models/model_final.pth \
    --downsample_refine_grid=True \
    --log_wandb=False \
	--edit_mask_thresh=${3} \
    --num_obj_voxels_thresh=${4} \
    --top_k_edit_thresh=${5} \
    --top_k_obj_thresh=${6}

	#python edit_pretrained_relu_field.py \
	#-d ./data/${1}/ \
	#-o logs/rf/${1}/${3}/ \
	#-i logs/rf/${1}/ref_real_200/saved_models/model_final.pth \
	#-p "$2" \
	#-eidx ${4} \
	#--log_wandb=False \
	#--do_refinement=True \
	#--lr_freq=500 \
	#--lr_decay_start=4000 \
	#--lr_gamma=0.85 \
	#--learning_rate=0.0075 \
	#--density_correlation_weight=1500.0 \
	#--data_pose_mode=True \
    #--downsample_refine_grid=True \
	#--num_iterations_edit=4000 \
	#--hf_auth_token=${5}

	# Rendering Output Video:
	#echo "Starting Rendering..."
	#python render_sh_based_voxel_grid_attn2.py \
	#-i logs/rf/${1}/${3}/saved_models/model_final_refined.pth \
	#-ie logs/rf/${1}/${3}/saved_models/model_final_attn_edit.pth \
	#-io logs/rf/${1}/${3}/saved_models/model_final_attn_object.pth \
	#-p "$2" \
	#-eidx "$4" \
	#--save_freq=10 \
	#--hf_auth_token=${5} \
	#-o output_renders/${1}/${3}_attn_wgt/

	## Rendering Output Video:
	#echo "Starting Rendering..."
	#python render_sh_based_voxel_grid.py \
	#-i logs/rf/${1}/${3}/saved_models/model_final.pth \
	#-o output_renders/${1}/${3}_unrefined/

	echo "Starting Rendering..."
	python render_sh_based_voxel_grid_attn.py \
	-i ./logs/rf/${1}/${2}/saved_models/model_final_refined.pth \
	-o output_renders/${1}/${2}/refined/ \
	--save_freq=10 
}

# STARTING RUN:

scene=pinecone
log_name="pineapple5"
edit_mask_thresh=1.0
num_obj_voxels_thresh=150000
top_k_edit_thresh=300
top_k_obj_thresh=30000


train_default $scene $log_name $edit_mask_thresh $num_obj_voxels_thresh $top_k_edit_thresh $top_k_obj_thresh

#scene=flowers
#prompt="a photo of a vase of sunflowers on the ground in a backyard"
#log_name="sunflowers"
#eidx=7
#hf_auth_token=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
#edit_mask_thresh=0.999
#num_obj_voxels_thresh=9000


#train_default $scene "$prompt" $log_name $eidx $hf_auth_token $edit_mask_thresh $num_obj_voxels_thresh