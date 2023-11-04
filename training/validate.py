import ipdb
from typing import List

import numpy as np
from requests.exceptions import ConnectionError
import torch
from PIL import Image
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import is_wandb_available
from tqdm import tqdm
from transformers import CLIPTokenizer

import matplotlib.pyplot as plt
from training.config import RunConfig
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.xti_attention_processor import XTIAttenProc
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from torchvision.utils import make_grid
from training import inference_dtu

if is_wandb_available():
    import wandb


class ValidationHandler:

    def __init__(self,
                 cfg: RunConfig,
                 placeholder_view_tokens: List[str],
                 placeholder_view_token_ids: List[int],
                 placeholder_object_tokens: List[str],
                 placeholder_object_token_ids: List[int],
                 fixed_object_token: str,
                 weights_dtype: torch.dtype,
                 max_rows: int = 14,
                 ):

        self.cfg = cfg
        self.placeholder_view_tokens = placeholder_view_tokens
        self.placeholder_view_token_ids = placeholder_view_token_ids
        self.placeholder_object_tokens = placeholder_object_tokens
        self.placeholder_object_token_ids = placeholder_object_token_ids
        self.fixed_object_token = fixed_object_token
        self.weight_dtype = weights_dtype
        self.max_rows = max_rows

        if len(placeholder_view_tokens) > 0:
            self.is_dtu = 'dtu' in placeholder_view_tokens[0]
        else:
            self.is_dtu = False

        if self.cfg.data.dtu_preprocess_key == 0:
            self.width, self.height = 512, 512
        elif self.cfg.data.dtu_preprocess_key == 1:
            # self.width, self.height = 512, 384
            self.width, self.height = 768, 576  # diferent inf dimension
        else:
            self.width, self.height = None, None


    def infer_dtu(self, accelerator, tokenizer, text_encoder, unet, vae,
                  num_images_per_prompt, seeds, step, prompts, return_instead_of_save=False):
        """
        return_instead_of_save=False
        return_instead_of_save: if True, then don't do any model logging.
        """
        train_cfg = self.cfg
        if self.cfg.debug:
            num_denoising_steps = 1
        else:
            num_denoising_steps = train_cfg.eval.num_denoising_steps

        ## get list of train and test view idxs, load gt images
        # table from camidx to gt images.
        cam_idxs, cam_idxs_train, cam_idxs_test = inference_dtu.get_cam_idxs(
            train_cfg.data.dtu_subset)

        ## Get image predictions, as a lookup from camidx->img.
        lookup_camidx_to_img_pred, pipeline = inference_dtu.dtu_generate_camidxs_to_preds(
            train_cfg=train_cfg,
            cam_idxs=cam_idxs,
            step=step,
            num_denoising_steps=num_denoising_steps,
            seeds=seeds,
            torch_dtype=self.weight_dtype,
            return_pipeline=True)

        if len(lookup_camidx_to_img_pred) == 0:
            # this means there was a connection error thrown by diffusers library
            # see handling in dtu_generate_camidxs_to_preds
            return

        # save the image predictions
        if not return_instead_of_save:
            fname_saved_images = Path(
            train_cfg.log.exp_dir
        ) / f"validation-iter_{step}-denoisesteps_{train_cfg.eval.num_denoising_steps}_numseeds_{len(seeds)}_upsample_{train_cfg.eval.dtu_upsample_key}.pt"
            torch.save(lookup_camidx_to_img_pred, fname_saved_images)
        assert set(lookup_camidx_to_img_pred.keys()) == set(cam_idxs)

        # collect gt images, as lookup camidx->img
        lookup_camidx_to_img_gt = inference_dtu.dtu_get_gt_images(
            cam_idxs, train_cfg.data.train_data_dir,
            train_cfg.data.dtu_lighting, train_cfg.data.dtu_preprocess_key)

        # collect mask, as lookup camidx-> img
        scan_id = Path(train_cfg.data.train_data_dir).stem[4:]
        lookup_camidx_to_mask = inference_dtu.get_object_masks(
            cam_idxs, scan_id)

        # do preprocessing -> torch tensors, to size (300,400)
        imgs_pred_all_seeds, imgs_gt, masks, imgs_gt, imgs_gt_plot = inference_dtu.process_imgs(
            cam_idxs, cam_idxs_train, lookup_camidx_to_img_pred,
            lookup_camidx_to_img_gt, lookup_camidx_to_mask)

        # run the numbers and get the grid
        results = inference_dtu.get_result_metrics_and_grids(
            cam_idxs, cam_idxs_train, imgs_pred_all_seeds, imgs_gt, masks, 
            imgs_gt_plot, seeds)

        # return early if not saving to loggers or files: this branch is called by inference.py
        if return_instead_of_save:
            return pipeline, results

        # save the images to the loggers
        for i, seed in enumerate(seeds):
            fname_saved_images = fname_saved_images.parent / f"{fname_saved_images.stem}_seed_{seed}.png"
            results['figures'][i].savefig(fname_saved_images, dpi=350)
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    pass
                if tracker.name == "wandb":
                    tracker.log({f"val_{i}": results['figures'][i]})
                    tracker.log

        for tracker in accelerator.trackers:
            log_metrics = {
                k: v
                for k, v in results.items()
                if k in ('mse_train_mean', 'mse_test_mean', 'psnr_train_mean',
                         'psnr_test_mean', 'ssim_train_mean', 'ssim_test_mean',
                         'lpips_train_mean', 'lpips_test_mean')
            }
            tracker.log(log_metrics)
        return pipeline, None

    def infer_mode3(self, accelerator, tokenizer, text_encoder, unet, vae,
                    num_images_per_prompt, seeds, step, prompts):
        """ similar to `infer_dtu` except we sample several objects.
        This has too many pipeline loads - should be once only
        """
        train_cfg = self.cfg
        if train_cfg.debug:
            num_denoising_steps = 1
        else:
            num_denoising_steps = train_cfg.eval.num_denoising_steps

        ## get list of train and test view idxs, load gt images
        # table from camidx to gt images.
        cam_idxs, cam_idxs_train, cam_idxs_test = inference_dtu.get_cam_idxs(
            train_cfg.data.dtu_subset)

        # do eval for each prompted placeholder tokens
        for i, eval_placeholder_object_token in enumerate(
                train_cfg.eval.eval_placeholder_object_tokens):

            # if train_cfg.debug and i>0:
            #     return pipeline

            lookup_camidx_to_img_pred, pipeline = inference_dtu.dtu_generate_camidxs_to_preds(
                train_cfg=train_cfg,
                eval_placeholder_object_token=eval_placeholder_object_token,
                cam_idxs=cam_idxs,
                step=step,
                num_denoising_steps=num_denoising_steps,
                seeds=[0],
                torch_dtype=self.weight_dtype,
                return_pipeline=True)

            fname_saved_images = Path(
                train_cfg.log.exp_dir
            ) / f"validation-iter_{step}-denoisesteps_{train_cfg.eval.num_denoising_steps}_objecttoken_{eval_placeholder_object_token}_upsample_{train_cfg.eval.dtu_upsample_key}.pt"
            torch.save(lookup_camidx_to_img_pred, fname_saved_images)
            # lookup_camidx_to_img_pred = torch.load(fname_saved_images)
            # save it to some dir
            assert set(lookup_camidx_to_img_pred.keys()) == set(cam_idxs)

            # lookup which dataset is linked to that eval_placeholder_token (for gt images)
            lookup_placeholder_token_to_subset = dict(
                zip(train_cfg.data.placeholder_object_tokens,
                    train_cfg.data.train_data_subsets))
            train_data_subset = lookup_placeholder_token_to_subset[
                eval_placeholder_object_token]
            train_data_subset = train_cfg.data.train_data_dir / train_data_subset

            ## collect gt images, as lookup camidx->img
            lookup_camidx_to_img_gt = inference_dtu.dtu_get_gt_images(
                cam_idxs, train_data_subset, train_cfg.data.dtu_lighting,
                train_cfg.data.dtu_preprocess_key)

            # collect mask, as lookup camidx-> img
            scan_id = train_data_subset.stem[4:]
            lookup_camidx_to_mask = inference_dtu.get_object_masks(
                cam_idxs, scan_id)

            # do preprocessing -> torch tensors, to size (300,400)
            imgs_pred_all_seeds, imgs_gt, masks, imgs_gt_plot = inference_dtu.process_imgs(
                cam_idxs, cam_idxs_train, lookup_camidx_to_img_pred,
                lookup_camidx_to_img_gt, lookup_camidx_to_mask)

            # run the numbers and get the grid
            results = inference_dtu.get_result_metrics_and_grids(
                cam_idxs, cam_idxs_train, imgs_pred_all_seeds, imgs_gt, masks,
                imgs_gt_plot, [0])

            fname_save_this = fname_saved_images.parent / f"{fname_saved_images.stem}_imgs_recon.png"
            f = results['figures'][0]
            f.savefig(fname_save_this, dpi=350)
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    pass
                if tracker.name == "wandb":
                    tracker.log({f"val_recon_{i}": f})

        # test text-to-image view control for the 34 DTU views. 
        # by default, DO_T2I_GENERALIZATION==False, since it takes a while and is not the main goal.
        DO_T2I_GENERALIZATION = False
        if DO_T2I_GENERALIZATION:
            t2i_prompts = [
                "a koala", "a brown teddy bear", "a small red car", "a small townhouse",
                "3 cans of soup", "a black dog"
            ]
            for i, t2i_prompt in enumerate(t2i_prompts):
                lookup_camidx_to_t2i_prompts = inference_dtu.dtu_generate_camidxs_to_preds(
                    train_cfg=train_cfg,
                    eval_placeholder_object_token=t2i_prompt,
                    cam_idxs=cam_idxs,
                    step=step,
                    num_denoising_steps=num_denoising_steps,
                    seeds=[0],
                    torch_dtype=self.weight_dtype,
                    return_pipeline=False)

                imgs = np.concatenate(
                    [lookup_camidx_to_t2i_prompts[idx] for idx in cam_idxs])
                imgs = torch.tensor(imgs).permute(0, 3, 1, 2)
                nrow = len(imgs)

                # as a reference, add in the gt images from one of the subsets
                train_data_dir = train_cfg.data.train_data_dir / train_cfg.data.train_data_subsets[
                    0]
                lookup_imgs_gt = inference_dtu.dtu_get_gt_images(
                    cam_idxs, train_data_dir, train_cfg.data.dtu_lighting,
                    train_cfg.data.dtu_preprocess_key)
                imgs_gt = np.stack([lookup_imgs_gt[idx] for idx in cam_idxs])
                imgs_gt = torch.tensor(imgs_gt).permute(0, 3, 1, 2)

                grid = make_grid(torch.cat((imgs, imgs_gt)),
                                 nrow=nrow)[:, ::2, ::2]
                f, axs = plt.subplots(figsize=(nrow, 3))
                axs.imshow(grid.permute(1, 2, 0))
                axs.set_axis_off()
                axs.set(title=t2i_prompt)
                fname_save_this = Path(
                    train_cfg.log.exp_dir
                ) / f"validation-iter_{step}-denoisesteps_{train_cfg.eval.num_denoising_steps}_upsample_{train_cfg.eval.dtu_upsample_key}_imgs_t2i_{i}.png"
                f.savefig(fname_save_this, dpi=350)

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        pass
                    if tracker.name == "wandb":
                        tracker.log({f"val_t2i_{i}": f})
        return pipeline

    def infer_disentangled_objects_dtu(self, pipeline, accelerator, tokenizer,
                                       text_encoder, unet, vae,
                                       num_images_per_prompt, seeds, step,
                                       prompts):
        """ 
        calls the pipeline loader specific to dtu that knows how to handle
        the novel tokens etc
        """
        train_cfg = self.cfg
        num_denoising_steps = 1 if train_cfg.debug else self.cfg.eval.num_denoising_steps

        if self.cfg.data.placeholder_object_tokens is None:
            placeholder_object_tokens = [
                self.cfg.data.placeholder_object_token
            ]
        else:
            placeholder_object_tokens = self.cfg.data.placeholder_object_tokens
        # if the dataset is too big, don't get stuck doing this
        if len(placeholder_object_tokens) > 10:
            placeholder_object_tokens = placeholder_object_tokens[::3]
            placeholder_object_tokens = placeholder_object_tokens[:10] 

        prompts = [f"A photo of a {t}" for t in placeholder_object_tokens]

        prompt_manager = PromptManager(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            timesteps=pipeline.scheduler.timesteps,
            placeholder_view_token_ids=self.placeholder_view_token_ids,
            placeholder_object_token_ids=self.placeholder_object_token_ids,
        )
        joined_images = []
        width, height = None, None
        if train_cfg.data.dtu_preprocess_key == 1:
            # width, height = (512,384)
            width, height = (768, 576)

        all_imgs = []
        for prompt in prompts:
            prompt_images = inference_dtu.run_inference(
                prompt=prompt,
                pipeline=pipeline,
                prompt_manager=prompt_manager,
                seeds=train_cfg.eval.validation_seeds,
                output_path=None,
                num_denoising_steps=num_denoising_steps,
                num_images_per_prompt=1,
                width=width,
                height=height,
                truncation_idx=None)
            nrow = len(prompt_images)
            prompt_images = torch.tensor(np.stack(prompt_images))
            all_imgs.append(prompt_images)

        all_imgs = torch.cat(all_imgs).permute(0, 3, 1, 2)
        grid = make_grid(all_imgs, nrow=nrow)
        f, axs = plt.subplots()
        axs.imshow(grid.permute(1, 2, 0))
        axs.set_axis_off()
        axs.set(title=placeholder_object_tokens)
        fname_save_this = Path(
            train_cfg.log.exp_dir
        ) / f"validation-iter_{step}-denoisesteps_{train_cfg.eval.num_denoising_steps}_upsample_{train_cfg.eval.dtu_upsample_key}_imgs_placeholder_object_tokens.png"
        f.savefig(fname_save_this, dpi=350)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                pass
            if tracker.name == "wandb":
                tracker.log({f"object_tokens": f})

    def infer(self,
              accelerator: Accelerator,
              tokenizer: CLIPTokenizer,
              text_encoder: NeTICLIPTextModel,
              unet: UNet2DConditionModel,
              vae: AutoencoderKL,
              num_images_per_prompt: int,
              seeds: List[int],
              step: int,
              prompts: List[str] = None):
        """ Runs inference during our training scheme. """
        if self.cfg.learnable_mode == 3:
            pipeline = self.infer_mode3(accelerator, tokenizer, text_encoder,
                                        unet, vae, num_images_per_prompt,
                                        seeds, step, prompts)

            self.infer_disentangled_objects_dtu(pipeline, accelerator,
                                                tokenizer, text_encoder, unet,
                                                vae, num_images_per_prompt,
                                                seeds, step, prompts)
            if self.cfg.debug:
                pass
                # ipdb.set_trace()
            return None

        if self.is_dtu:
            pipeline, _ = self.infer_dtu(accelerator, tokenizer, text_encoder,
                                      unet, vae, num_images_per_prompt, seeds,
                                      step, prompts)

            self.infer_disentangled_objects_dtu(pipeline, accelerator,
                                                tokenizer, text_encoder, unet,
                                                vae, num_images_per_prompt,
                                                seeds, step, prompts)

            return None

        try:
            pipeline = self.load_stable_diffusion_model(
                accelerator, tokenizer, text_encoder, unet, vae)
        except ConnectionError as e:
            try:
                sleep(60 * 5)
                pipeline = self.load_stable_diffusion_model(
                    accelerator, tokenizer, text_encoder, unet, vae)
            except ConnectionError as e:
                print("Connection error, resuming")
                print(e)
                return None

        prompt_manager = PromptManager(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            timesteps=pipeline.scheduler.timesteps,
            placeholder_view_token_ids=self.placeholder_view_token_ids,
            placeholder_object_token_ids=self.placeholder_object_token_ids,
        )

        if prompts is None:
            if self.cfg.learnable_mode == 0:
                assert len(self.placeholder_object_tokens) == 1
                prompts = [
                    p.format(self.placeholder_object_tokens[0])
                    for p in self.cfg.eval.validation_prompts
                ]

            elif self.cfg.learnable_mode in (1, 2, 4, 5):
                if self.cfg.eval.validation_view_tokens is not None:
                    view_tokens = self.cfg.eval.validation_view_tokens
                else:
                    view_tokens = self.placeholder_view_tokens

                if len(view_tokens) > 100:
                    view_tokens = view_tokens[::30].copy()
                if self.is_dtu and len(view_tokens) > 15:
                    view_tokens = view_tokens[::3]

                view_tokens = view_tokens[:self.max_rows - 1].copy()

                if self.cfg.learnable_mode == 1:
                    prompts = [
                        f"{v}. A photo of a {self.fixed_object_token}"
                        for v in view_tokens
                    ]
                else:
                    prompts = [
                        f"A photo of a {self.placeholder_object_tokens[0]}"
                    ]
                    prompts += [
                        f"{v}. A photo of a {self.placeholder_object_tokens[0]}"
                        for v in view_tokens
                    ]

            else:
                raise

        else:
            prompts = prompts

        joined_images = []

        for prompt in prompts:
            images = self.infer_on_prompt(
                pipeline=pipeline,
                prompt_manager=prompt_manager,
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                seeds=seeds)
            prompt_image = Image.fromarray(np.concatenate(images, axis=1))
            joined_images.append(prompt_image)

        final_image = Image.fromarray(np.concatenate(joined_images, axis=0))
        final_image.save(self.cfg.log.exp_dir / f"val-image-{step}.png")
        try:
            self.log_with_accelerator(accelerator,
                                      joined_images,
                                      step=step,
                                      prompts=prompts)
        except:
            pass
        del pipeline
        torch.cuda.empty_cache()
        if text_encoder.text_model.embeddings.mapper_view is not None:
            text_encoder.text_model.embeddings.mapper_view.train()
        # change dict to ModuleDIct
        if text_encoder.text_model.embeddings.mapper_object_lookup is not None:
            for mapper in text_encoder.text_model.embeddings.mapper_object_lookup.values():
                mapper.train()

        # for mapper in (text_encoder.text_model.embeddings.mapper_object,
        #                text_encoder.text_model.embeddings.mapper_view):
            if mapper is not None:
                mapper.train()
        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        return final_image

    def infer_on_prompt(self,
                        pipeline: StableDiffusionPipeline,
                        prompt_manager: PromptManager,
                        prompt: str,
                        seeds: List[int],
                        num_images_per_prompt: int = 1) -> List[Image.Image]:
        prompt_embeds = self.compute_embeddings(prompt_manager=prompt_manager,
                                                prompt=prompt)
        all_images = []
        for idx in tqdm(range(num_images_per_prompt)):
            generator = torch.Generator(device='cuda').manual_seed(seeds[idx])
            images = sd_pipeline_call(
                pipeline,
                prompt_embeds=prompt_embeds,
                generator=generator,
                num_inference_steps=self.cfg.eval.num_denoising_steps,
                num_images_per_prompt=1,
                height=self.height,
                width=self.width).images
            all_images.extend(images)
        return all_images

    @staticmethod
    def compute_embeddings(prompt_manager: PromptManager,
                           prompt: str) -> torch.Tensor:
        with torch.autocast("cuda"):
            with torch.no_grad():
                prompt_embeds = prompt_manager.embed_prompt(prompt)
        return prompt_embeds

    def load_stable_diffusion_model(
            self, accelerator: Accelerator, tokenizer: CLIPTokenizer,
            text_encoder: NeTICLIPTextModel, unet: UNet2DConditionModel,
            vae: AutoencoderKL) -> StableDiffusionPipeline:
        """ Loads SD model given the current text encoder and our mapper. """
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            torch_dtype=self.weight_dtype)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.scheduler.set_timesteps(self.cfg.eval.num_denoising_steps,
                                         device=pipeline.device)
        pipeline.unet.set_attn_processor(XTIAttenProc())
        if text_encoder.text_model.embeddings.mapper_view is not None:
            text_encoder.text_model.embeddings.mapper_view.eval()
        # change dict to ModuleDIct
        if text_encoder.text_model.embeddings.mapper_object_lookup is not None:
            for mapper in text_encoder.text_model.embeddings.mapper_object_lookup.values():
                mapper.eval()
        return pipeline

    def log_with_accelerator(self, accelerator: Accelerator,
                             images: List[Image.Image], step: int,
                             prompts: List[str]):
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation",
                                          np_images,
                                          step,
                                          dataformats="NHWC")
            if tracker.name == "wandb":

                tracker.log({
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                })
