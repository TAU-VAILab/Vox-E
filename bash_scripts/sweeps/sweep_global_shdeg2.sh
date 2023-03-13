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
	--density_correlation_weight=150 \
	--data_downsample_factor=4.0 \
	--parallel_rays_chunk_size=16384 \
	--feedback_frequency=10000 \
	--ray_batch_size=45000 \
	--render_num_samples_per_ray=128 \
	--tv_density_weight=50.0 \
	--tv_features_weight=150.0 \
	--learning_rate=0.028 \
	--do_refinement=False \
	--sds_t_start=4000 \
	--sh_degree=2 # we currently only support diffuse

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

sweep_name=sweep_shdeg2
scene=dog2
prompt="a render of a yarn doll of a light grey dog"
log_name="yarn"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_shdeg2
scene=gingercat
prompt="a render of a yarn doll of a ginger cat"
log_name="yarn"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_shdeg2
scene=kangaroo
prompt="a render of a yarn doll of a kangaroo"
log_name="yarn"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_shdeg2
scene=dog2
prompt="a render of a wood carving of a light grey dog"
log_name="wood"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_shdeg2
scene=gingercat
prompt="a render of a wood carving of a ginger cat"
log_name="wood"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_shdeg2
scene=kangaroo
prompt="a render of a wood carving of a kangaroo"
log_name="wood"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_shdeg2
scene=dog2
prompt="a render of a claymation figure of a light grey dog"
log_name="claymation"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_shdeg2
scene=gingercat
prompt="a render of a claymation figure of a ginger cat"
log_name="claymation"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx

sweep_name=sweep_shdeg2
scene=kangaroo
prompt="a render of a claymation figure of a kangaroo"
log_name="claymation"
eidx=8

train_default $scene $sweep_name "$prompt" $log_name $eidx


