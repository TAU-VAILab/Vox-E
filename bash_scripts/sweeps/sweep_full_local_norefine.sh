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
	-o logs/rf/${2}/${1}/${4} \
	-i logs/rf/${2}/${1}/ref/saved_models/model_final.pth \
	-p "$3" \
	-eidx=${5} \
	--num_iterations_edit=8000 \
	--directional_dataset=True \
	--density_correlation_weight=50 \
	--tv_density_weight=50.0 \
	--tv_features_weight=100.0 \
	--learning_rate=0.025 \
	--do_refinement=False \
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

# christmas sweater

sweep_name=sweep_full_local_norefine
scene=gingercat
prompt="a render of a cat wearing a christmas sweater"
log_name="christmas"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_norefine
scene=dog2
prompt="a render of a dog wearing a christmas sweater"
log_name="christmas"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_norefine
scene=kangaroo
prompt="a render of a kangaroo wearing a christmas sweater"
log_name="christmas"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

# sunglasses

sweep_name=sweep_full_local_norefine
scene=gingercat
prompt="a render of a cat wearing big sunglasses"
log_name="sunglasses"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_norefine
scene=kangaroo
prompt="a render of a kangaroo wearing big sunglasses"
log_name="sunglasses"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_norefine
scene=dog2
prompt="a render of a dog wearing a big sunglasses"
log_name="sunglasses"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

# birthday hat

sweep_name=sweep_full_local_norefine
scene=gingercat
prompt="a render of a cat wearing a birthday hat"
log_name="birthday"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_norefine
scene=dog2
prompt="a render of a dog wearing a birthday hat"
log_name="birthday"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_norefine
scene=kangaroo
prompt="a render of a kangaroo wearing a birthday hat"
log_name="birthday"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx