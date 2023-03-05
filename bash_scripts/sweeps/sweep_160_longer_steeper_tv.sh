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
	--sds_t_start=6000 \
	--sds_t_gamma=0.7 \
	--sds_t_freq=600 \
	--num_iterations_per_stage=10000 \
	--lr_gamma=0.96 \
	--tv_density_weight=20.0 \
	--tv_features_weight=40.0 \
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

#sweep_name=sweep_160_longer_steeper_tv
#scene=taxi
#prompt="a render of a taxi made of legos"
#log_name="taxilego"
#
#train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=dog2
prompt="a render of a light grey minecraft dog"
log_name="minecraft_lightgrey"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=gingercat
prompt="a render of a cat wearing a santa hat"
log_name="santa"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=gingercat
prompt="a render of a ginger minecraft cat"
log_name="ginger_minecraft"
train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=dog2
prompt="a render of a light grey dog made of legos"
log_name="legos_lightgrey"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=dog2
prompt="a render of a dog wearing big sunglasses"
log_name="bigglasses"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=gingercat
prompt="a render of a cat wearing a magician's hat"
log_name="magicianhat"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=kangaroo
prompt="a render of a kangaroo on rollerskates"
log_name="rollerskates"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=kangaroo
prompt="a render of a pink balloon in the shape of a  kangaroo"
log_name="pink_balloon"

train_default $scene $sweep_name "$prompt" $log_name

sweep_name=sweep_160_longer_steeper_tv
scene=taxi
prompt="a render of a yarn doll of a car"
log_name="yarn"

train_default $scene $sweep_name "$prompt" $log_name
