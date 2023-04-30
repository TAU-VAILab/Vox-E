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
	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}/ref/saved_models/model_final.pth \
	-o output_renders/${2}/${1}/${4}/ \
	--sds_prompt="$3" \
	--save_freq=10
}

# STARTING RUN:

sweep_name=sweep_full_local_extended
scene=dog1
prompt="a render of a dog"
log_name="inputs"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_extended
scene=cat2
prompt="a render of a cat"
log_name="inputs"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_local_extended
scene=alien
prompt="a render of an alien"
log_name="inputs"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_extended
scene=dog1
prompt="a render of a dog"
log_name="inputs"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_extended
scene=cat2
prompt="a render of a cat"
log_name="inputs"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_full_global_extended
scene=alien
prompt="a render of an alien"
log_name="inputs"
eidx=9

train_default $scene $sweep_name "$prompt" $log_name $eidx
