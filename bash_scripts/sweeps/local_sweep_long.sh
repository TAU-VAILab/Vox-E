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
train_default() {
	# Train:
	python run_sds_on_high_res_model.py \
	-d ../data/${1}/ \
	-o logs/rf/${2}/${1}/${4} \
	-i logs/rf/${2}/${1}/ref/saved_models/model_final.pth \
	-p "$3" \
	--directional_dataset=True \
	--density_correlation_weight=200 \
	--sds_t_start=5000 \
	--sds_t_gamma=0.75 \
	--sds_t_freq=800 \
	--num_iterations_per_stage=10000 \
	--sh_degree=0 # we currently only support diffuse

	# Rendering Output Video:
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${2}/${1}/${4}/saved_models/model_final.pth \
	-o output_renders/${2}/${1}/${4}/ \
	--sds_prompt="$3" \
	--save_freq=10
}

# STARTING RUN:
sweep_name="local_sweep_long"
scene=gingercat
prompt="a render of a cat wearing a magician's hat"
log_name="magician hat"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name="local_sweep_long"
scene=gingercat
prompt="a render of a cat with a huge smile"
log_name="smile"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name="local_sweep_long"
scene=kangaroo
prompt="a render of a kangaroo on rollerskates"
log_name="rollerskates"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name="local_sweep_long"
scene=kangaroo
prompt="a render of a kangaroo wearing a santa hat"
log_name="santa"

train_default $scene $sweep_name "$prompt" $log_name
