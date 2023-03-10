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
	--density_correlation_weight=500 \
	--tv_density_weight=50.0 \
	--tv_features_weight=100.0 \
	--learning_rate=0.025 \
	--do_refinement=False \
	--uncoupled_mode=True \
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

# lego

sweep_name=sweep_full_global_uncoupled
scene=gingercat
prompt="a render of a cat made of legos"
log_name="lego"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=dog2
prompt="a render of a dog made of legos"
log_name="lego"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=kangaroo
prompt="a render of a kangaroo made of legos"
log_name="lego"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=taxi
prompt="a render of a yellow car made of legos"
log_name="lego"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=frog2
prompt="a render of a frog made of legos"
log_name="lego"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

# sunglasses

sweep_name=sweep_full_global_uncoupled
scene=gingercat
prompt="a render of a yarn doll of a cat"
log_name="yarn"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=kangaroo
prompt="a render of a yarn doll of a kangaroo"
log_name="yarn"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=dog2
prompt="a render of a yarn doll of a dog"
log_name="yarn"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=taxi
prompt="a render of a yarn doll of yellow car"
log_name="yarn"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=frog2
prompt="a render of a yarn doll of a frog"
log_name="yarn"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

# birthday hat

sweep_name=sweep_full_global_uncoupled
scene=gingercat
prompt="a render of a minecraft cat"
log_name="minecraft"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=dog2
prompt="a render of a minecraft dog"
log_name="minecraft"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=kangaroo
prompt="a render of a minecraft kangaroo"
log_name="minecraft"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=taxi
prompt="a render of a yellow minecraft car"
log_name="minecraft"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_uncoupled
scene=frog2
prompt="a render of a minecraft frog"
log_name="minecraft"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx