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
train_and_render_attn_grid() {
	# Train:
	echo "Starting Training..."
	python run_attn_grid_on_sds_model.py \
	-d ../data/${1}/ \
	--sds_model_path=logs/rf/${1}/${3}/saved_models/model_final.pth \
	-o logs/rf/${1}/${3}/attn/ \
	--prompt="$2" \
	--edit_idx=${4} \
    --object_idx=${5} \
	--timestamp=${6}

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid_attn.py \
	-i logs/rf/${1}/${3}/attn/saved_models/model_final_edit.pth \
	-o output_renders/attn/${1}/${3}/edit/

	python render_sh_based_voxel_grid_attn.py \
	-i logs/rf/${1}/${3}/attn/saved_models/model_final_object.pth \
	-o output_renders/attn/${1}/${3}/object/
}
# STARTING RUN:

#scene=gingercat
#prompt="a render of a ginger cat wearing a magician's hat"
#log_name="magicianhat" # 1-word description of the prompt for saving
#edit_idx=11
#object_idx=6
#timestamp=200
#
#train_and_render_attn_grid $scene "$prompt" $log_name $edit_idx $object_idx $timestamp

#scene=dog2
#prompt="a render of a dog wearing big sunglasses"
#log_name="sunglasses" # 1-word description of the prompt for saving
#edit_idx=8
#object_idx=5
#timestamp=200
#
#train_and_render_attn_grid $scene "$prompt" $log_name $edit_idx $object_idx $timestamp

scene=dog2
prompt="a render of a dog wearing a christmas sweater"
log_name="christmas2" # 1-word description of the prompt for saving
edit_idx=9
object_idx=5
timestamp=200

train_and_render_attn_grid $scene "$prompt" $log_name $edit_idx $object_idx $timestamp