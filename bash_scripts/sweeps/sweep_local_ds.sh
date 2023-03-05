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
	python run_sds_on_high_res_model.py \
	-d ../data/${1}/ \
	-o logs/rf/${2}/${1}/${4} \
	-i logs/rf/${2}/${1}/ref/saved_models/model_final.pth \
	-p "$3" \
	--directional_dataset=True \
	--density_correlation_weight=200 \
	--sds_t_start=4000 \
	--sds_t_gamma=0.75 \
	--sds_t_freq=600 \
	--num_iterations_per_stage=8000 \
	--tv_density_weight=130.0 \
	--tv_features_weight=100.0 \
	--sh_degree=0 # we currently only support diffuse

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${2}/${1}/${4}/saved_models/model_final.pth \
	-o output_renders/${2}/${1}/${4}/ \
	--ref_path=logs/rf/${2}/${1}/ref/saved_models/model_final.pth \
	--sds_prompt="$3" \
	--save_freq=10
}

# STARTING RUN:

# earmuffs

#sweep_name=sweep_local_ds
#scene=gingercat
#prompt="a render of a cat wearing earmuffs"
#log_name="earmuffs"
#
#train_default $scene $sweep_name "$prompt" $log_name
#
#sweep_name=sweep_local_ds
#scene=dog2
#prompt="a render of a dog wearing earmuffs"
#log_name="earmuffs"
#
#train_default $scene $sweep_name "$prompt" $log_name
#
#sweep_name=sweep_local_ds
#scene=kangaroo
#prompt="a render of a kangaroo wearing earmuffs"
#log_name="earmuffs"
#
#train_default $scene $sweep_name "$prompt" $log_name

# christmas sweater

#sweep_name=sweep_local_ds
#scene=gingercat
#prompt="a render of a cat wearing a christmas sweater"
#log_name="christmas"
#
#train_default $scene $sweep_name "$prompt" $log_name
#
#sweep_name=sweep_local_ds
#scene=dog2
#prompt="a render of a dog wearing a christmas sweater"
#log_name="christmas"
#
#train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_local_ds
scene=kangaroo
prompt="a render of a kangaroo wearing a christmas sweater"
log_name="christmas"

train_default $scene $sweep_name "$prompt" $log_name

# sunglasses

#sweep_name=sweep_local_ds
#scene=gingercat
#prompt="a render of a cat wearing a scarf"
#log_name="scarf"
#
#train_default $scene $sweep_name "$prompt" $log_name
#
#sweep_name=sweep_local_ds
#scene=dog2
#prompt="a render of a dog wearing a scarf"
#log_name="scarf"
#
#train_default $scene $sweep_name "$prompt" $log_name
#
#sweep_name=sweep_local_ds
#scene=kangaroo
#prompt="a render of a kangaroo wearing a scarf"
#log_name="scarf"
#
#train_default $scene $sweep_name "$prompt" $log_name

# sunglasses

sweep_name=sweep_local_ds
scene=gingercat
prompt="a render of a cat wearing big sunglasses"
log_name="sunglasses"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_local_ds
scene=dog2
prompt="a render of a dog wearing big sunglasses"
log_name="sunglasses"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_local_ds
scene=kangaroo
prompt="a render of a kangaroo wearing big sunglasses"
log_name="sunglasses"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_local_ds
scene=gingercat
prompt="a render of a cat wearing a magician's hat"
log_name="magician"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_local_ds
scene=dog2
prompt="a render of a dog wearing a magician's hat"
log_name="magician"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_local_ds
scene=kangaroo
prompt="a render of a kangaroo wearing a magician's hat"
log_name="magician"

train_default $scene $sweep_name "$prompt" $log_name