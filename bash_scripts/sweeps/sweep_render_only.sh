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
train_and_render() {
	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}_sds_dir_${3}_dcl_${5}_${4}_sds_t_${6}_${7}_${8}/saved_models/model_final.pth \
	-o output_renders/${9}/${1}/${4} \
	--sds_prompt="$2" \
	--save_freq=10
}

# STARTING RUN:

scene=gingercat
prompt="a render of a claymation figure of a cat"
directional=True
log_name="claymation" # 1-word description of the prompt for saving
dcl_weight=200.0
sds_t_decay_start=4000
sds_t_gamma=0.75
sds_t_freq=500
sweep_name=$sweep_name_in

train_and_render $scene "$prompt" $directional $log_name $dcl_weight $sds_t_decay_start \
$sds_t_gamma $sds_t_freq $sweep_name