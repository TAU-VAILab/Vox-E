# SDSReluFields

## Concept:
![Graph](https://tau-vailab.github.io/Vox-E/images/overview_official.png "Flow:")

## Getting the Repo:
    git clone https://github.com/etaisella/voxelArtReluFields.git
    cd voxelArtReluFields

</br>

## Setting up Conda Environment:
    conda create --name <env_name> --python=3.10
    conda activate <env_name>
    pip install -r requirements.txt

</br>

## Getting the Data:
Download "sds_relu_fields_data.zip" from [here](https://drive.google.com/drive/folders/15nsQQzF1ykgefZ4WXuINgdOM90VxtXvL?usp=sharing).
Unzip it in the repo folder.

</br>

## Running The Demo:
    # Make a "regular" ReLU field:
    bash bash_scripts/train_default_relu_field.sh -d dog2

    # Run SDS training on it:
    bash bash_scripts/run_sds_on_high_res_model.py




