import clip
import pandas as pd
import click
import torch
import os
from easydict import EasyDict
from torch.backends import cudnn
from pathlib import Path
from PIL import Image

# Age-old custom option for fast training :)
cudnn.benchmark = True
# Also set torch's multiprocessing start method to spawn
# refer -> https://github.com/pytorch/pytorch/issues/40403
# for more information. Some stupid PyTorch stuff to take care of
torch.multiprocessing.set_start_method("spawn")

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
        
        # get ref images
        ref_path = os.path.join(scene_dir, 'ref')
        ref_imgs = get_images(ref_path)
        clip_ref_img_features = get_CLIP_im_features(ref_imgs)
        clip_ref_text_features = get_text_features(ref_path)

        # create dataframe
        prompts = []
        out_to_text_sims = []
        directional_sims = []
        
        # iterate over prompt dirs
        for prompt_dir_name in os.listdir(scene_dir):
            # skip ref folder, you've been there
            if prompt_dir_name == 'ref':
                continue
            prompts.append(prompt_dir_name)
            prompt_dir = os.path.join(scene_dir, prompt_dir_name)
            clip_out_text_features = get_text_features(prompt_dir)
            output_imgs = get_images(prompt_dir)
            clip_out_img_features = get_CLIP_im_features(output_imgs)

            # record [Output - Text] CliP similarity
            out_text_similarity = get_avg_CLIP_text_sim(clip_out_img_features, clip_out_text_features)
            out_to_text_sims.append(out_text_similarity)

            # record directional CliP similarity
            directional_similarity = get_avg_CLIP_directional_sim(clip_ref_text_features, \
                                                                  clip_ref_img_features, \
                                                                  clip_out_text_features, \
                                                                  clip_out_img_features)
            directional_sims.append(directional_similarity)
        
        # set up dataframe
        metrics_dict = {'out img to prompt similarity': out_to_text_sims,
                        'directional similarity': directional_sims}
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
    
    
def get_images(ref_path: Path) -> tuple:
    ims = []
    for name in img_names_to_get:
        im_path = os.path.join(ref_path, name)
        img = Image.open(im_path)
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


if __name__ == "__main__":
    main()
