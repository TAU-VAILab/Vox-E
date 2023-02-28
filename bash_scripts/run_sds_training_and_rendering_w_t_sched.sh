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
	-o logs/rf/${1}_sds_dir_${3}_dcl_${5}_${4}_sds_t_${6}_${7}_${8}/ \
	-i logs/rf/${1}_ref_shdeg_0/saved_models/model_final.pth \
	-p "$2" \
	--directional_dataset=${3} \
	--density_correlation_weight=${5} \
	--sds_t_start=${6} \
	--sds_t_gamma=${7} \
	--sds_t_freq=${8} \
	--sh_degree=0 # we currently only support diffuse

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}_sds_dir_${3}_dcl_${5}_${4}_sds_t_${6}_${7}_${8}/saved_models/model_final.pth \
	-o output_renders/${1}_sds_dir_${3}_dcl_${5}_${4}_sds_t_${6}_${7}_${8} \
	--save_freq=10
}

# STARTING RUN:
scene=kangaroo
prompt="a render of a kangaroo wearing big sunglasses"
directional=True
log_name="bigglasses" # 1-word description of the prompt for saving
dcl_weight=200.0
sds_t_decay_start=4000
sds_t_gamma=0.75
sds_t_freq=500

train_and_render $scene "$prompt" $directional $log_name $dcl_weight $sds_t_decay_start $sds_t_gamma $sds_t_freq