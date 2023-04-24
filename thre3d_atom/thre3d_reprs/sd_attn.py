import cv2
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
import matplotlib.pyplot as plt

from thre3d_atom.thre3d_reprs.cross_attn import AttentionStore

# suppress partial model loading warning
logging.set_verbosity_error()
import thre3d_atom.thre3d_reprs.cross_attn as ca
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)

        # dummy loss value
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        use_auth_token = False
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == '1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
            use_auth_token = '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'
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", use_auth_token=use_auth_token).to(
            self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", use_auth_token=use_auth_token)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder",
                                                          use_auth_token=use_auth_token).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", use_auth_token=use_auth_token).to(
            self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", use_auth_token=use_auth_token)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, self.device)
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]
        batch_size = len(prompt)

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([""] * batch_size, padding='max_length',
                                      max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def get_attn_map(self, prompt, pred_rgb, timestamp=0, indices_to_alter=[7], guidance_scale=100,  logvar=None):
        batch_size = len(prompt)
        controller = ca.AttentionStore()
        ca.register_attention_control(self.unet, controller)
        # interp to 512x512 to be fed into vae.

        with torch.no_grad():
            orig_im_h, orig_im_w = pred_rgb.shape[-2:]
            text_embeddings = self.get_text_embeds(prompt, '')
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
            if timestamp > 0:
                t = torch.as_tensor(timestamp, dtype=torch.long, device=self.device)
            latents = self.encode_imgs(pred_rgb_512)
            latents = latents.expand(batch_size, self.unet.in_channels, 512 // 8, 512 // 8).to(self.device)
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = controller.step_callback(latents)

            attn_maps = None
            if indices_to_alter is not None:
                attn_maps = ca.aggregate_and_get_max_attention_per_token(
                    prompts=prompt,
                    attention_store=controller,
                    indices_to_alter=indices_to_alter, orig_im_h=orig_im_h, orig_im_w=orig_im_h
                )
        return attn_maps[0], t.item()



    def train_step(self, dir, indices_to_alter, prompts_dir, text_embeddings, pred_rgb, blend_word, guidance_scale=100,
                   logvar=None, extra_noise=False, extra_attn='', use_controllers=False, input_idx=0, controllers=None):
        batch_size = len(prompts_dir[dir])
        lb = None
        if len(blend_word) > 0:
            lb = ca.LocalBlend(self.tokenizer, prompts_dir[dir], (blend_word, blend_word), device=self.device)
        controller = ca.AttentionRefine(self.tokenizer, prompts_dir[dir], self.num_train_timesteps,
                                        cross_replace_steps=1.,
                                        self_replace_steps=1., local_blend=lb, device=self.device)
        if len(extra_attn) > 0:
            controller_a = controller
            equalizer = ca.get_equalizer(self.tokenizer, prompts_dir[dir][1], (extra_attn,), (5,))
            controller = ca.AttentionReweight(self.tokenizer, prompts_dir[dir], self.num_train_timesteps,
                                              cross_replace_steps=1.,
                                              self_replace_steps=1., equalizer=equalizer, local_blend=lb,
                                              controller=controller_a)
        if use_controllers:
            if input_idx in controllers:
                controller = controllers[input_idx]
            else:
                controllers[input_idx] = controller
        ca.register_attention_control(self.unet, controller)
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        orig_im_h, orig_im_w = pred_rgb.shape[-2:]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        latents = latents.expand(batch_size, self.unet.in_channels, 512 // 8, 512 // 8).to(self.device)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            attn_maps = None
            if indices_to_alter is not None:
                attn_maps = ca.aggregate_and_get_max_attention_per_token(
                    prompts=prompts_dir[dir],
                    attention_store=controller,
                    indices_to_alter=indices_to_alter, orig_im_h=orig_im_h, orig_im_w=orig_im_h
                )
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        if extra_noise:
            denoised = latents.clone().detach()
        latents = controller.step_callback(latents)
        if extra_noise:
            noise_pred = noise_pred + (denoised - latents)

            # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)

        # if global_step % 101 == 1 or global_step == 1:
        #     attn = show_cross_attention(self.tokenizer, prompts_dir[dir], controller, res=16,
        #                                 from_where=("up", "mid", "down"))
        #     a = 1
        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        if logvar != None:
            grad = grad * torch.exp(-1 * logvar)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # _t = time.time()
        loss = SpecifyGradient.apply(latents, grad)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')
        # image = latent2image(self.vae, latents)
        # attn = show_cross_attention(self.tokenizer, prompt, controller, res=16, from_where=("up", "down"))
        return loss, attn_maps

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'],
                        help="stable diffusion version")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()


class scoreDistillationLoss(nn.Module):
    def __init__(self,
                 device,
                 prompts,
                 directional=False, blend_word=''):
        super().__init__()
        self.dir_to_indx_dict = {}
        self.directional = directional
        # get sd model
        # self.sd_model = StableDiffusion(device,"2.0", hf_key="Fictiverse/Stable_Diffusion_VoxelArt_Model")
        self.sd_model = StableDiffusion(device, "1.4")
        self.blend_word = blend_word
        self.controllers = {}
        # encode text
        if directional:
            self.text_encodings = {}
            self.prompts = {}
            for dir_prompt in ['side', 'overhead', 'back', 'front']:
                print(f"Encoding text for \'{dir_prompt}\' direction")
                modified_prompt = [prompt + f", {dir_prompt} view" for prompt in prompts] if \
                    isinstance(prompts, list) else prompts + f", {dir_prompt} view"
                self.prompts[dir_prompt] = modified_prompt if isinstance(prompts, list) else [modified_prompt]
                self.text_encodings[dir_prompt] = self.sd_model.get_text_embeds(modified_prompt, '')
            # self.sd_model.set_ca(self.prompts)
        else:
            self.text_encoding = self.sd_model.get_text_embeds(prompts, '')

    def training_step(self, indices_to_alter, output, image_height, image_width, global_step, directions=None,
                      logvars=None, extra_noise=False, extra_attn='', controllers=False, input_idx=0):
        loss = 0
        if self.directional:
            assert (directions != None), f"Must supply direction if SDS loss is set to directional mode"
        # format output images
        out_imgs = torch.reshape(output, (-1, image_height, image_width, 3))
        out_imgs = out_imgs.permute((0, 3, 1, 2))

        # perform training step
        if not self.directional:
            loss = self.sd_model.train_step(global_step, self.text_encoding, out_imgs, logvar=logvars)
        else:
            for idx, dir_prompt in enumerate(directions):
                if logvars != None:
                    logvar = logvars[idx]
                else:
                    logvar = None
                encoding = self.text_encodings[dir_prompt]
                loss_sds, attn_maps = self.sd_model.train_step(dir_prompt, indices_to_alter, self.prompts, encoding,
                                                               out_imgs, self.blend_word,
                                                               logvar=logvar, extra_noise=extra_noise,
                                                               extra_attn=extra_attn, use_controllers=controllers, input_idx=input_idx,
                                                               controllers=self.controllers)
                loss = loss + loss_sds

        return loss, attn_maps
