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
train_and_render() {
	# Train:
	echo "Starting Training..."
	python run_sds_on_high_res_model.py \
	-d ../data/${1}/ \
	-o logs/rf/${1}_sds_dir_${3}_dcl_${5}_${4}_front_overhead/ \
	-i logs/rf/high_res_${1}_diffuse/saved_models/model_final.pth \
	-p "$2" \
	--directional_dataset=${3} \
	--density_correlation_weight=${5} \
	--new_frame_frequency=1 \
	--sh_degree=0 # we currently only support diffuse

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}_sds_dir_${3}_dcl_${5}_${4}_front_overhead/saved_models/model_final.pth \
	-o output_renders/${1}_sds_dir_${3}_dcl_${5}_${4}_front_overhead
}

# STARTING RUN:


scene=frog
prompt="a render of a cute green frog wearing mittens"
directional=True
log_name="mittens" # 1-word description of the prompt for saving
dcl_weight=1500.0

train_and_render $scene "$prompt" $directional $log_name $dcl_weight

scene=frog
prompt="a render of a cute olive green tree frog with a big smile"
directional=True
log_name="bigsmile" # 1-word description of the prompt for saving
dcl_weight=1500.0

train_and_render $scene "$prompt" $directional $log_name $dcl_weight

scene=lego
prompt="a yellow toy bulldozer riding a blue magic carpet"
directional=True
log_name="carpet" # 1-word description of the prompt for saving
dcl_weight=1500.0

train_and_render $scene "$prompt" $directional $log_name $dcl_weight

scene=hotdog
prompt="a yarn doll of two hotdogs on a plate"
directional=True
log_name="yarndoll" # 1-word description of the prompt for saving
dcl_weight=1500.0

train_and_render $scene "$prompt" $directional $log_name $dcl_weight