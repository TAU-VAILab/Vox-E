# Vox-E: Text-guided Voxel Editing of 3D Objects

This is the official pytorch implementation of Vox-E.

[![arXiv](https://img.shields.io/badge/arXiv-2303.12048-b31b1b.svg)](https://arxiv.org/abs/2303.12048)
![Generic badge](https://img.shields.io/badge/conf-ICCV2023-purple.svg)

[[Project Website](https://tau-vailab.github.io/Vox-E/)]

> **Vox-E: Text-guided Voxel Editing of 3D Objects**<br>
> Etai Sella<sup>1</sup>, Gal Fiebelman <sup>1</sup>, Peter Hedman<sup>2</sup>, Hadar Averbuch-Elor<sup>1</sup><br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>Google Research

>**Abstract** <br>
> Large scale text-guided diffusion models have garnered significant attention due to their ability to synthesize diverse images that convey complex visual concepts. This generative power has more recently been leveraged to perform text-to-3D synthesis. In this work, we present a technique that harnesses the power of latent diffusion models for editing existing 3D objects. Our method takes oriented 2D images of a 3D object as input and learns a grid-based volumetric representation of it. To guide the volumetric representation to conform to a target text prompt, we follow unconditional text-to-3D methods and optimize a Score Distillation Sampling (SDS) loss. However, we observe that combining this diffusion-guided loss with an image-based regularization loss that encourages the representation not to deviate too strongly from the input object is challenging, as it requires achieving two conflicting goals while viewing only structure-and-appearance coupled 2D projections. Thus, we introduce a novel volumetric regularization loss that operates directly in 3D space, utilizing the explicit nature of our 3D representation to enforce correlation between the global structure of the original and edited object. Furthermore, we present a technique that optimizes cross-attention volumetric grids to refine the spatial extent of the edits. Extensive experiments and comparisons demonstrate the effectiveness of our approach in creating a myriad of edits which cannot be achieved by prior works.

![Graph](https://tau-vailab.github.io/Vox-E/images/voxe_teaser.png "Flow:")
</br>

# Getting Started

## Getting the repo
    git clone https://github.com/etaisella/Vox-E.git
    cd Vox-E

</br>

## Setting up conda environment
    conda create --name vox-e python=3.10.10 --yes
    conda activate vox-e
    pip install -r requirements.txt

</br>

## Getting the data
For the synthetic scenes shown in the paper (and used in the demo) download "voxe_data.zip" from [here](https://drive.google.com/file/d/1h1X3NppS4V2PtCHg4gSCaZO93ZvFqnRO/view?usp=sharing) 
and unzip in the repo folder. 
</br>
Other datasets shown in the paper include:
- The synthetic NeRF dataset, which can be found [here](https://drive.google.com/drive/u/0/folders/1-iJug5cTJA7bhDnhIxTraH5EyuyRA7sr).
- The NeRF 360 real-scenes, which can be found [here](https://drive.google.com/file/d/18hxar-OXpHA_SuX2pYsVkad0sltdK5GK/view?usp=sharing).


</br>

## Running The Demo
We first need to learn a default reconstruction grid depicting the scene. </br>
This is done by running the following bash script and supplying it with a scene name (in the demo's case - "dog2").

    bash bash_scripts/train_default_relu_field.sh -d dog2

Now we can perform textual edits on our reconstruction grid.</br>
We offer two demos, one which performs a global edit (i.e. making the dog into a yarn doll) and one which performs a local edit (i.e. putting a party hat on the dog). </br>
Local edits in our pipeline are refined using an additional refinement stage. Note that this makes the execution time for local edits longer.</br>

To run the global edit demo, run:

    bash bash_scripts/edit_demo_global.sh

To run the local edit demo there is a requirement to add a Hugging Face authentication token (instructions for how to get
the token can be found [here](https://huggingface.co/docs/hub/security-tokens)) run:

    bash bash_scripts/edit_demo_local.sh -a <Hugging Face authentication token>
</br>

When finished you should see a 360 rendering video of the edited output in:
Local edit demo:
    Vox-E/output_renders/dog2/party_hat/rendered_video.mp4

Global edit demo:
    Vox-E/output_renders/dog2/yarn/rendered_video.mp4

</br>

### 'Real-Scenes' demo
We have also included a demo which edits a scene from ['360 real-scenes'](https://drive.google.com/file/d/18hxar-OXpHA_SuX2pYsVkad0sltdK5GK/view?usp=sharing). 
After extracting the contents of the dataset zip file into the repo folder (resulting in a Vox-E/nerf_360/ folder structure), navigate to the repo folder and follow the steps below to run the demo:
 </br>

Reconstruct the Pinecone scene:

	bash bash_scripts/real_scenes/train_default_relu_field_real.sh -d pinecone

 Run the edit demo:

 	bash bash_scripts/real_scenes/edit_demo_real.sh -a <Hugging Face authentication token>

</br>

### Tested Configuration
This code has been tested with Python 3.10.10, PyTorch 1.13.0, CUDA 11.4 on Ubuntu 20.04.5 LTS. </br>
Execution (for the editing stage) takes around 50 minutes on an NVIDIA RTX A5000 GPU. 

</br>

# General Usage

## Initialization

To learn an initial feature grid which reconstructs the input scene run:

    python train_sh_based_voxel_grid_with_posed_images.py -d <path to dataset folder>
                                                          -o <path to output folder>
                                                          --sh_degree 0

This step is pre-requirement for performing editing.

</br>

## Editing

To run our system and perform textual edits on 3D scenes run:

    python edit_pretrained_relu_field.py -d <path to dataset folder>
                                         -o <path to output folder>
                                         -i <path to initial feature grid>
                                         -p <text prompt>

To perform local edits and use the refinement stage, three additional arguments are required:

    --do_refinement True
    --edit_idx <index of edit token>
    --hf_auth_token <Hugging Face authentication token>

The edit index is the index of the token associated with the edit word in the text prompt (for example the word "hat" in the prompt - "a dog wearing a hat"). </br>
To find this token for your text prompt we recommend using [this](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite) 
huggingface space.

We offer also an additonal post-processing stage for denoising by retaining the largest connected component, to perform
this exra stage add the following argument:

	--post_process_scc True

</br>

## BibTeX
If you find our work useful in your research, please consider citing:

    @article{sella2023vox,
     title={Vox-E: Text-guided Voxel Editing of 3D Objects},
     author={Sella, Etai and Fiebelman, Gal and Hedman, Peter and Averbuch-Elor, Hadar},
     journal={arXiv preprint arXiv:2303.12048},
     year={2023}
    }
    
</br>

# Acknowledgements

We thank [Animesh Karnewar](https://akanimax.github.io/) for his wonderful ReLU-Fields code on which we base our own.
We also thank the "printable_models" user on [free3d](https://free3d.com/) for creating many of the meshes we use as data.
