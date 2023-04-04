import clip
import pandas as pd
import click
import torch
from torch import Tensor
import os
import pytorch_fid.fid_score as fid
from easydict import EasyDict
from torch.backends import cudnn
from pathlib import Path
from PIL import Image
import math
from typing import Any
import torchvision.transforms
from torch.nn.functional import l1_loss, mse_loss
INFINITY = 1e10

def mse2psnr(x: Any) -> Any:
    if isinstance(x, Tensor):
        dtype, device = x.dtype, x.device
        # fmt: off
        return (
            -10.0 * torch.log(x) / torch.log(torch.tensor([10.0], dtype=dtype, device=device))
            if x != 0.0
            else torch.tensor([INFINITY], dtype=dtype, device=device)
        )
        # fmt: on
    else:
        return -10.0 * math.log(x) / math.log(10.0) if x != 0.0 else math.inf

# Age-old custom option for fast training :)
cudnn.benchmark = True
# Also set torch's multiprocessing start method to spawn
# refer -> https://github.com/pytorch/pytorch/issues/40403
# for more information. Some stupid PyTorch stuff to take care of

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_names_to_get = ["0.png", "20.png", "140.png"]

# load clip
print(f"Available CLiP models - {clip.available_models()}")
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()

# Required arguments:
@click.option("-d", "--result_folder", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path to result folder")

def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)
    result_path = Path(config.result_folder)

    # make dataframe array
    dataframes = []
    frame_titles = []
    
    # iterate over scene subdirs:
    for scene_dir_name in os.listdir(result_path):
        scene_dir = os.path.join(result_path, scene_dir_name)


        # checking if it is a file
        if os.path.isfile(scene_dir):
            continue
        
        remove_word_from_filenames(scene_dir, "color_")

        # get ref images
        recon_path = os.path.join(scene_dir, 'recon')
        recon_imgs = get_images(recon_path)
        clip_recon_img_features = get_CLIP_im_features(recon_imgs)
        clip_input_features = get_text_features(recon_path)

        # get input path
        input_path = os.path.join(scene_dir, 'inputs')
        input_imgs = get_images(input_path)

        # create dataframe
        prompts = []
        out_to_text_sims = []
        directional_sims = []
        fid_recon_scores = []
        fid_input_scores = []
        psnr_refs = []
        
        # iterate over prompt dirs
        for prompt_dir_name in os.listdir(scene_dir):
            # skip ref folder, you've been there
            if (prompt_dir_name == 'inputs') or (prompt_dir_name == 'recon'):
                continue
            prompts.append(prompt_dir_name)
            prompt_dir = os.path.join(scene_dir, prompt_dir_name)
            clip_out_text_features = get_text_features(prompt_dir)
            fid_score_recon = fid.calculate_fid_given_paths((prompt_dir, recon_path),
                                                   50,
                                                   device,
                                                   2048,
                                                   1)
            fid_score_input = fid.calculate_fid_given_paths((prompt_dir, input_path),
                                                   50,
                                                   device,
                                                   2048,
                                                   1)
            fid_recon_scores.append(fid_score_recon)
            fid_input_scores.append(fid_score_input)
            output_imgs = get_images(prompt_dir)

            # get psnr
            psnr_recon = get_PSNRS(output_imgs, recon_imgs)
            psnr_refs.append(psnr_recon)

            clip_out_img_features = get_CLIP_im_features(output_imgs)

            # record [Output - Text] CliP similarity
            out_text_similarity = get_avg_CLIP_text_sim(clip_out_img_features, clip_out_text_features)
            out_to_text_sims.append(out_text_similarity)

            # record directional CliP similarity
            directional_similarity = get_avg_CLIP_directional_sim(clip_input_features, \
                                                                  clip_recon_img_features, \
                                                                  clip_out_text_features, \
                                                                  clip_out_img_features)
            directional_sims.append(directional_similarity)
        
        # set up dataframe
        metrics_dict = {'text CLIP': out_to_text_sims,
                        'dir CLIP': directional_sims,
                        'FID recon': fid_recon_scores,
                        'FID input': fid_input_scores,
                        'PSNR recon': psnr_refs}
        df = pd.DataFrame(data=metrics_dict, index=prompts)
        frame_titles.append(scene_dir_name)
        dataframes.append(df)
    
    output_csv_path = os.path.join(result_path, "output_metrics.csv")
    with open(output_csv_path, 'w') as file:
        for df, df_title in zip(dataframes, frame_titles):
            file.write(df_title)
            file.write('\n')
            df.to_csv(file)
            file.write('\n')
        

def get_PSNRS(out_imgs: tuple, ref_imgs):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Resize((400,400))])
    total_PSNR = 0.0
    with torch.no_grad():
        for out_img, ref_img in zip(out_imgs, ref_imgs):
            out_t = transform(out_img).reshape((-1, 3)).to(device)
            ref_t = transform(ref_img).reshape((-1, 3)).to(device)
            mse = mse_loss(out_t, ref_t)
            psnr = mse2psnr(mse)
            total_PSNR += psnr
    return (total_PSNR.item() / len(out_imgs))


def get_avg_CLIP_directional_sim(ref_txt_features: torch.Tensor, \
                                 ref_img_features: torch.Tensor, \
                                 out_txt_features: torch.Tensor, \
                                 out_img_features: torch.Tensor) -> float:
    total_sim = 0.0
    ref_txt_feat_normed = ref_txt_features / ref_txt_features.norm(dim=-1, keepdim=True)
    out_txt_feat_normed = out_txt_features / out_txt_features.norm(dim=-1, keepdim=True)
    text_dir = ref_txt_feat_normed - out_txt_feat_normed

    for out_im_feat, ref_im_feat in zip(out_img_features, ref_img_features):
        ref_im_feat_normed = ref_im_feat / ref_im_feat.norm(dim=-1, keepdim=True)
        out_im_feat_normed = out_im_feat / out_im_feat.norm(dim=-1, keepdim=True)
        im_dir = ref_im_feat_normed - out_im_feat_normed
        sim = (text_dir @ im_dir.T).item()
        total_sim += sim

    return total_sim / len(out_img_features)


def get_avg_CLIP_text_sim(out_features: tuple, text_features: torch.Tensor) -> float:
    total_sim = 0.0
    target_feat_normed = text_features / text_features.norm(dim=-1, keepdim=True)
    for out_feat in out_features:
        out_feat_normed = out_feat / out_feat.norm(dim=-1, keepdim=True)
        sim = (out_feat_normed @ target_feat_normed.T).item()
        total_sim += sim
    return total_sim / len(out_features)


def get_text_features(prompt_path: Path) -> str:
    prompt_filepath = os.path.join(prompt_path, "prompt.txt")
    with open(prompt_filepath, 'r') as file:
        prompt = file.readlines()[0]  
    text = clip.tokenize(prompt).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features
    
    
def get_images(im_dir: Path) -> tuple:
    ims = []
    for name in os.listdir(im_dir):
        if not name.endswith('.png'):
            continue
        im_path = os.path.join(im_dir, name)
        img = Image.open(im_path).convert("RGB") 
        ims.append(img)
    return ims


def get_CLIP_im_features(imgs: tuple) -> tuple:
    im_features = []
    for img in imgs:
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img)
        im_features.append(image_features)
    return im_features

# thanks chatGPT :)
def remove_word_from_filenames(folder_path, word_to_remove):
    """
    Recursively iterates over a folder and removes a given word from filenames within the folders.

    Args:
        folder_path (str): The path to the folder to iterate over.
        word_to_remove (str): The word to remove from filenames.

    Returns:
        None
    """
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if word_to_remove in filename:
                # Construct the new filename without the word to remove.
                new_filename = os.path.join(root, filename).replace(word_to_remove, "")
                # Rename the file.
                os.rename(os.path.join(root, filename), new_filename)


if __name__ == "__main__":
    main()
