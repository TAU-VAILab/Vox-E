# Vox-E: Text-guided Voxel Editing of 3D Objects

[![arXiv](https://img.shields.io/badge/arXiv-2303.12048-b31b1b.svg)](https://arxiv.org/abs/2303.12048)

[[Project Website](https://tau-vailab.github.io/Vox-E/)]

> **Vox-E: Text-guided Voxel Editing of 3D Objects**<br>
> Etai Sella<sup>1</sup>, Gal Fiebelman <sup>1</sup>, Peter Hedman<sup>2</sup>, Hadar Averbuch-Elor<sup>1</sup><br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>Google Research

>**Abstract**: <br>
> Large scale text-guided diffusion models have garnered significant attention due to their ability to synthesize diverse images that convey complex visual concepts. This generative power has more recently been leveraged to perform text-to-3D synthesis. In this work, we present a technique that harnesses the power of latent diffusion models for editing existing 3D objects. Our method takes oriented 2D images of a 3D object as input and learns a grid-based volumetric representation of it. To guide the volumetric representation to conform to a target text prompt, we follow unconditional text-to-3D methods and optimize a Score Distillation Sampling (SDS) loss. However, we observe that combining this diffusion-guided loss with an image-based regularization loss that encourages the representation not to deviate too strongly from the input object is challenging, as it requires achieving two conflicting goals while viewing only structure-and-appearance coupled 2D projections. Thus, we introduce a novel volumetric regularization loss that operates directly in 3D space, utilizing the explicit nature of our 3D representation to enforce correlation between the global structure of the original and edited object. Furthermore, we present a technique that optimizes cross-attention volumetric grids to refine the spatial extent of the edits. Extensive experiments and comparisons demonstrate the effectiveness of our approach in creating a myriad of edits which cannot be achieved by prior works.

![Graph](https://tau-vailab.github.io/Vox-E/images/overview_official.png "Flow:")
</br>

## Description:
This is the official pytorch implementation of Vox-E.
</br>
</br>

# Getting Started:

## Getting the repo:
    git clone https://github.com/etaisella/Vox-E.git
    cd Vox-E

</br>

## Setting up conda environment:
    conda create --name <env_name> --python=3.10
    conda activate <env_name>
    pip install -r requirements.txt

</br>

## Getting the data:
Download "voxe_data.zip" from [here](https://drive.google.com/file/d/1h1X3NppS4V2PtCHg4gSCaZO93ZvFqnRO/view?usp=sharing) 
and unzip in the repo folder. 
</br>
Note that in the paper and the project page we also showcase results on the NeRF dataset, which can be found 
[here](https://drive.google.com/drive/u/0/folders/1-iJug5cTJA7bhDnhIxTraH5EyuyRA7sr).

</br>

## Running The Demo:
We first need to learn a default reconstruction grid depicting the scene. </br>
This is done by running the following bash script and supplying it with a scene name (in the demo's case - "dog2").

    bash bash_scripts/train_default_relu_field.sh -d dog2

Now we can perform textual edits on our reconstruction grid.</br>
We offer two demos, one which performs a global edit (i.e. making the dog into a yarn doll) and one which performs a local edit (i.e. putting a party hat on the dog). </br>
Local edits in our pipeline are refined using an additional refinement stage. Note that this makes the execution time for local edits longer.</br>

To run the global edit demo, run:

    bash bash_scripts/edit_demo_global.py

To run the local edit demo, run:

    bash bash_scripts/edit_demo_local.py

</br>

# General Usage:

## Initialization:

To learn an initial feature grid which reconstructs the input scene run:

    python train_sh_based_voxel_grid_with_posed_images.py -d <path to dataset folder>
                                                          -o <path to output folder>
                                                          --sh_degree 0

This step is pre-requirement for performing editing.

## Editing:

To run our system and perform textual edits on 3D scenes run:

    python edit_pretrained_relu_field.py -d <path to dataset folder>
                                         -o <path to output folder>
                                         -i <path to initial feature grid>
                                         -p <text prompt>

To perform local edits and use the refinement stage, two additional arguments are required:

    --do_refinement True
    --edit_idx <index of edit token>

The edit index is the index of the token associated with the edit word in the text prompt (for example the word "hat" in the prompt - "a dog wearing a hat"). </br>
To find this token for your text prompt we recommend using [this](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite) 
huggingface space.

</br>

# Acknowledgements:

We thank [Animesh Karnewar](https://akanimax.github.io/) for his wonderful ReLU-Fields code on which we base our own.
We also thank the "printable_models" user on [free3d](https://free3d.com/) for creating many of the meshes we use as data.

