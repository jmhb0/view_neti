""" Common methods for dealing with DTU in eval """
import ipdb
import sys
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pyrallis
import torch
from PIL import Image, ImageOps
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPTokenizer
from pathlib import Path
from torchvision.utils import make_grid
from lpips import LPIPS
import torch
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation
from skimage import transform
import torchvision.transforms.functional as T_f
import torchvision.transforms as T

sys.path.append(".")
sys.path.append("..")

import constants
import tqdm
from constants import IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_SMALL, PATH_DTU_CALIBRATION_DIR, DTU_SPLIT_IDXS, DTU_MASKS
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from models.xti_attention_processor import XTIAttenProc
from checkpoint_handler import CheckpointHandler
from utils import vis_utils
from training import inference_dtu
from training.config import RunConfig
from training.dataset import TextualInversionDataset
from utils.utils import num_to_string, string_to_num


def get_cam_idxs(dtu_subset):
    """ 
    Get list of train and test view idxs for dtu_subset in ('3','6','9')
    """
    # the 'train' saved to DTU_SPLIT_IDXS is correct only for subset '9'.
    cam_idxs = DTU_SPLIT_IDXS['train'] + DTU_SPLIT_IDXS['test']
    cam_idxs = sorted(cam_idxs)  # don't change ordering after this call
    cam_idxs_train = TextualInversionDataset.dtu_get_train_idxs(dtu_subset)
    cam_idxs_test = [idx for idx in cam_idxs if idx not in cam_idxs_train]

    return cam_idxs, cam_idxs_train, cam_idxs_test


def dtu_get_gt_images(cam_idxs, train_data_dir, dtu_lighting,
                      dtu_preprocess_key):
    """
    `dtu_preprocess_key` determines the dimensions we'll use for resizing.
    """
    fnames_gt = [
        Path(train_data_dir) / TextualInversionDataset.dtu_cam_and_lighting_to_fname(
            idx, dtu_lighting) for idx in cam_idxs
    ]

    lookup_camidx_to_gt_image = {}
    for (idx, f) in zip(cam_idxs, fnames_gt):
        image = Image.open(f)

        if dtu_preprocess_key == 0:
            padding = (0, 0, 0, 400)
            image = ImageOps.expand(image, padding, fill='black')
            assert image.size == (1600, 1600)
            image = image.resize((768, 768), resample=Image.Resampling.BICUBIC)
        elif dtu_preprocess_key == 1:
            image = image.resize((768, 576))
        else:
            raise NotImplementedError

        lookup_camidx_to_gt_image[idx] = image

    return lookup_camidx_to_gt_image


def dtu_generate_camidxs_to_preds(train_cfg,
                                  cam_idxs,
                                  step,
                                  num_denoising_steps=30,
                                  seeds: List[int] = [0, 1],
                                  eval_placeholder_object_token=None,
                                  torch_dtype=torch.float16,
                                  generate_bigger_images=0,
                                  guidance_scale=7.5,
                                  return_pipeline=False):
    """ 
    From supplied cam_idxs from DTU, generate predictions (`len(seeds)` of them0)

    This code reloads a fresh stable-diffusion model, reloads the trained neti-
    mappers from files, and so on. 
    For use during training, this is possibly overkill, but it simplifies the 
    code a litte bit and it ensures that `inference` behavior matches training 
    eval behaviour. It also only adds probably 30seconds to each eval, which is 
    not tooooooo big compared to how long eval on DTU takes anyway. 


    train_cfg: config of the neti-mapper models that we'll load. 
    step (int): training step for the pretrained model. Used for getting the
        pretrained neti models. 

    eval_placeholder_object_token: which placeholder object token to use for eval. 
    If None then take it from train_cfg.cfg.placeholder_object_token; this option
    is standard for learnable_mode 2.
    """
    # paths to placeholder tokens and the Neti mappers for them.
    learned_embeds_path = train_cfg.log.exp_dir / f"learned_embeds-steps-{step}.bin"
    mapper_checkpoint_path_view, mapper_checkpoint_path_object = None, None
    if train_cfg.learnable_mode != 0:
        mapper_checkpoint_path_view = train_cfg.log.exp_dir / f"mapper-steps-{step}_view.pt"
    if train_cfg.learnable_mode != 1:
        mapper_checkpoint_path_object = train_cfg.log.exp_dir / f"mapper-steps-{step}_object.pt"

    ## load the pipeline and the placeholder tokens for the pretrained object
    print("loading pipeline")
    mapper = None  # dummy for now, but it will be updated later
    try:
        pipeline, placeholder_tokens, placeholder_token_ids = load_stable_diffusion_model(
            pretrained_model_name_or_path=train_cfg.model.
            pretrained_model_name_or_path,
            mapper=mapper,
            num_denoising_steps=num_denoising_steps,
            learned_embeds_path=learned_embeds_path,
            torch_dtype=torch_dtype)
    except ConnectionError as e:
        print("CONNECTION ERROR, SKIPPING THIS VAL LOOP")
        return {}

    ## split out the placeholders into types: views and objects
    (placeholder_view_tokens, placeholder_view_token_ids,
     placeholder_object_tokens,
     placeholder_object_token_ids) = split_placeholders(
         placeholder_tokens, placeholder_token_ids)

    ## generate maps from camidxs to params, fnames, and tokens for all possible cams in DTU
    # recover them as lookup tables (dicts)
    (lookup_camidx_to_view_token, lookup_camidx_to_cam_params
     ) = TextualInversionDataset.dtu_generate_dset_cam_tokens_params()

    ## identify tokens not already in the tokenizer, add them, resize embeddings, get token ids
    placeholder_view_tokens_new = [
        p for p in list(lookup_camidx_to_view_token.values())
        if p not in placeholder_view_tokens
    ]
    num_added_tokens = pipeline.tokenizer.add_tokens(
        placeholder_view_tokens_new)
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    placeholder_view_token_ids_new = [
        pipeline.tokenizer.convert_tokens_to_ids(t)
        for t in placeholder_view_tokens_new
    ]

    ## load pretrained neti mappers, object and view
    mapper_view, mapper_object = None, None
    # raise NotImplementedError("need to add object tokens to tokenizer, pass the tokenids to load_mapper",
    # " so they can setup the load_mapper_dict correctly, and I think that's it")
    if train_cfg.learnable_mode != 0:
        _, mapper_view = CheckpointHandler.load_mapper(
            mapper_checkpoint_path_view,
            embedding_type='view',
            placeholder_view_tokens=placeholder_view_tokens,
            placeholder_view_token_ids=placeholder_view_token_ids,
        )

    is_pretrained_object = '/' in train_cfg.data.fixed_object_token_or_path

    if train_cfg.learnable_mode not in range(6):
        raise NotImplementedError()
    if train_cfg.learnable_mode in (2, 3, 4, 5) or is_pretrained_object:
        if is_pretrained_object:
            fname = train_cfg.data.fixed_object_token_or_path
        elif train_cfg.learnable_mode in (2, 3, 4, 5):
            fname = mapper_checkpoint_path_object

        _, mapper_object_lookup = CheckpointHandler.load_mapper(
            fname,
            embedding_type='object',
            placeholder_object_tokens=placeholder_object_tokens,
            placeholder_object_token_ids=placeholder_object_token_ids)

    ## add the new tokens to the Neti-view-mapper. This isn't necessary for object mappers
    if mapper_view is not None:
        mapper_view.add_view_tokens_to_vocab(placeholder_view_tokens_new,
                                             placeholder_view_token_ids_new)

    ## add the mappers to the text encoder
    pipeline.text_encoder.text_model.embeddings.set_mapper(
        mapper_object_lookup, mapper_view)

    ## prepare the prompt manager that handles embeddings for placeholder tokens
    placeholder_view_token_ids_all = placeholder_view_token_ids + placeholder_view_token_ids_new
    placeholder_view_tokens_all = placeholder_view_tokens + placeholder_view_tokens_new

    prompt_manager = PromptManager(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        timesteps=pipeline.scheduler.timesteps,
        unet_layers=constants.UNET_LAYERS,
        placeholder_view_token_ids=placeholder_view_token_ids_all,
        placeholder_object_token_ids=placeholder_object_token_ids,
        torch_dtype=torch_dtype)

    ## create prompts for all images we want generations for (all cam_idxs)
    is_pretrained_object = '/' in train_cfg.data.fixed_object_token_or_path

    if train_cfg.learnable_mode not in range(6):
        raise NotImplementedError()
    if eval_placeholder_object_token:
        object_token = eval_placeholder_object_token
        if eval_placeholder_object_token not in placeholder_object_tokens:
            pipeline.text_encoder.text_model.embeddings.mapper_object_lookup = None

    elif is_pretrained_object or train_cfg.learnable_mode in (2, 4, 5):
        object_token = placeholder_object_tokens[0]  # only
    else:
        object_token = train_cfg.data.fixed_object_token_or_path

    assert object_token in placeholder_object_tokens

    prompts = [
        f"{lookup_camidx_to_view_token[idx]}. A photo of a {object_token}"
        for idx in cam_idxs
    ]

    ## do the image generation / novel view prediction
    lookup_camidx_to_imgs_pred = {}
    if train_cfg.data.dtu_preprocess_key == 1:
        # width, height = (512,384)
        width, height = (768, 576)

    # if generate_bigger_images: 
    #     width, height = (1152, 864)


    for i, (cam_idx, prompt) in tqdm.tqdm(enumerate(zip(cam_idxs, prompts)),
                                          total=len(prompts)):
        print(f"*** Prompt {i} of {len(prompts)}")
        prompt_images = run_inference(prompt=prompt,
                                      pipeline=pipeline,
                                      prompt_manager=prompt_manager,
                                      seeds=seeds,
                                      output_path=None,
                                      num_denoising_steps=num_denoising_steps,
                                      num_images_per_prompt=1,
                                      width=width,
                                      height=height,
                                      guidance_scale=guidance_scale,
                                      truncation_idx=None)
        lookup_camidx_to_imgs_pred[cam_idx] = np.stack(prompt_images)
        if 0:
            fname = exp_dir / "debug.png"
            grid = vis_utils.get_image_grid(prompt_images)
            grid.save(fname)

    if return_pipeline:
        return lookup_camidx_to_imgs_pred, pipeline
    else:
        return lookup_camidx_to_imgs_pred


def load_stable_diffusion_model(
    pretrained_model_name_or_path: str,
    learned_embeds_path: Path,
    mapper: Optional[NeTIMapper] = None,
    num_denoising_steps: int = 50,
    torch_dtype: torch.dtype = torch.float16
) -> Tuple[StableDiffusionPipeline, str, int]:
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              local_files_only=True,
                                              subfolder="tokenizer")
    text_encoder = NeTICLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=True,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    )
    if mapper is not None:
        raise NotImplementedError()
        # text_encoder.text_model.embeddings.set_mapper(mapper)

    # load the view tokens
    placeholder_tokens, placeholder_token_ids = CheckpointHandler.load_learned_embed_in_clip(
        learned_embeds_path=learned_embeds_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer)

    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=True,
        torch_dtype=torch_dtype,
        text_encoder=text_encoder,
        tokenizer=tokenizer).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_denoising_steps,
                                     device=pipeline.device)
    pipeline.unet.set_attn_processor(XTIAttenProc())
    return pipeline, placeholder_tokens, placeholder_token_ids


def split_placeholders(placeholder_tokens, placeholder_token_ids):
    """ 
    For lists of placeholder_tokens and corresponding placeholder_token_ids (this
    is what is output by `load_stable_diffusion_model`), split into object and 
    view subsets.
    """
    # extract the view and placeholder tokens based on their names
    idxs_view = [
        i for i, s in enumerate(placeholder_tokens) if s[:5] == "<view"
    ]
    placeholder_view_tokens = [placeholder_tokens[i] for i in idxs_view]
    placeholder_view_token_ids = [placeholder_token_ids[i] for i in idxs_view]
    idxs_object = [
        i for i, s in enumerate(placeholder_tokens) if s[:5] != "<view"
    ]
    placeholder_object_tokens = [placeholder_tokens[i] for i in idxs_object]
    placeholder_object_token_ids = [
        placeholder_token_ids[i] for i in idxs_object
    ]
    # assert len(placeholder_object_tokens) in (0, 1)

    return placeholder_view_tokens, placeholder_view_token_ids, placeholder_object_tokens, placeholder_object_token_ids


def run_inference(prompt: str,
                  pipeline: StableDiffusionPipeline,
                  prompt_manager: PromptManager,
                  seeds: List[int],
                  output_path: Optional[Path] = None,
                  height: int = None,
                  width: int = None,
                  num_images_per_prompt: int = 1,
                  num_denoising_steps=50,
                  guidance_scale=7.5,
                  truncation_idx: Optional[int] = None) -> List[Image.Image]:
    with torch.autocast("cuda"):
        with torch.no_grad():
            prompt_embeds = prompt_manager.embed_prompt(
                prompt,
                num_images_per_prompt=num_images_per_prompt,
                truncation_idx=truncation_idx)
    joined_images = []
    for seed in seeds:
        generator = torch.Generator(device='cuda').manual_seed(seed)
        images = sd_pipeline_call(
            pipeline,
            num_inference_steps=num_denoising_steps,
            prompt_embeds=prompt_embeds,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt).images
        seed_image = Image.fromarray(np.concatenate(images,
                                                    axis=1)).convert("RGB")
        if output_path is not None:
            save_name = f'{seed}_truncation_{truncation_idx}.png' if truncation_idx is not None else f'{seed}.png'
            seed_image.save(output_path / save_name)
        joined_images.append(seed_image)
    # joined_image = vis_utils.get_image_grid(joined_images)
    return joined_images


def get_object_masks(cam_idxs, scan_idx, dtu_preprocess_key=1):
    lookup_camidx_to_mask = {}

    for cam_idx in cam_idxs:
        dir_mask = Path(DTU_MASKS) / f"scan{scan_idx}/mask"

        # case 1: the mask is in the parent
        if not os.path.exists(dir_mask):
            f_mask = dir_mask.parent / f"{cam_idx:03d}.png"
        # case 2: the mask is in the child
        else:
            f_mask = dir_mask / f"{cam_idx:03d}.png"
        try:
            mask = Image.open(f_mask).convert("RGB")
        # if file doesn't exist, just get an all-white mask
        except FileNotFoundError:
            mask = Image.new("RGB", (1600, 1200), color=(255, 255, 255))

        if dtu_preprocess_key == 1:
            mask = mask.resize((400,300))

        lookup_camidx_to_mask[cam_idx] = mask

    return lookup_camidx_to_mask


def process_imgs(cam_idxs, cam_idxs_train, lookup_camidx_to_img_pred,
                 lookup_camidx_to_img_gt, lookup_camidx_to_mask):
    """ 
        Preprocess all image data: torch tensors with axes (...,C,H,W), range 
        [0,1] for imgs, (0,1) for maks. Resize to the standard inference dims 
        for dtu, (h,w)=(300,400).

        `imgs_pred` have k samples, for k seeds

        Output: 
            imgs_pred: (bs,n_seeds,c,h,w)
            imgs_gt: (bs,c,h,w)
            masks: (bs,c,h,w)
            imgs_plot: same as masks but with h slightly bigger for padding viz

        """
    # make everything torch tensors, check ndims
    imgs_pred = np.stack([lookup_camidx_to_img_pred[i] for i in cam_idxs])
    assert imgs_pred.ndim == 5, "expected (bs,n_seeds,h,w,3)"
    imgs_pred = torch.tensor(imgs_pred).permute(0, 1, 4, 2, 3)

    # ipdb.set_trace()
    imgs_gt = np.stack([lookup_camidx_to_img_gt[i] for i in cam_idxs])
    masks = np.stack([lookup_camidx_to_mask[i] for i in cam_idxs])
    assert imgs_gt.ndim == 4 and masks.ndim == 4, "expected (bs,h,w,3)"
    imgs_gt = torch.tensor(imgs_gt).permute(0, 3, 1, 2)
    masks = torch.tensor(masks).permute(0, 3, 1, 2)

    # check the aspect ratios are all good
    h_pred, w_pred = imgs_pred.shape[-2:]
    h_gt, w_gt = imgs_gt.shape[-2:]
    assert h_gt / w_gt == h_pred / w_pred == 0.75

    # resize to (300,400) inference dimension, which is standard in prior work
    h_new, w_new = 300, 400
    T_resize = T.Resize((h_new, w_new),
                        interpolation=T.InterpolationMode.BICUBIC)
    imgs_gt = T_resize(imgs_gt)
    masks = T_resize(masks)
    # imgs_pred needs special treatment bc of its extra batch dim
    bs, n_seeds, c, h, w = imgs_pred.shape
    imgs_pred = T_resize(imgs_pred.view(bs * n_seeds, c, h, w))
    imgs_pred = imgs_pred.contiguous().view(bs, n_seeds, c, h_new, w_new)

    # create a copy of gt with a header that is yellow for 'train' images
    imgs_gt_plot = []
    for i, cam_idx in enumerate(cam_idxs):
        if cam_idx in cam_idxs_train:
            header = torch.ones(
                (3, 50, w_new)) * torch.tensor([255, 255, 0])[:, None, None]
        else:
            header = torch.zeros((3, 50, w_new))
        imgs_gt_plot.append(
            torch.cat((header, imgs_gt[i]), dim=1).unsqueeze(0))
    imgs_gt_plot = torch.cat(imgs_gt_plot)

    # normalize
    imgs_pred = imgs_pred / 255.
    imgs_gt = imgs_gt / 255.
    imgs_gt_plot = imgs_gt_plot / 255.
    masks = masks / 255.
    thresh = 0.01
    masks[masks > thresh] = 1
    masks[masks <= thresh] = 0

    return imgs_pred, imgs_gt, masks, imgs_gt, imgs_gt_plot


def get_result_metrics_and_grids(cam_idxs,
                                 cam_idxs_train,
                                 imgs_pred_all_seeds,
                                 imgs_gt,
                                 masks,
                                 imgs_gt_plot,
                                 seeds,
                                 do_lpips=False,
                                 title_prefix=""):
    """ 
    Generate summary metrics and plots. 
    Recommend setting `do_lpips=True` only during separate eval bc I've had some 
    OOM errors.   """
    is_cam_idx_train = torch.tensor(
        [idx in cam_idxs_train for idx in cam_idxs])

    # compute metrics over all the random seeds and average
    mse_train, psnr_train, ssim_train, lpips_train = [], [], [], []
    mse_test, psnr_test, ssim_test, lpips_test = [], [], [], []

    # iterate over the random seeds
    grids, figures, all_imgs_pred, all_imgs_gt = [], [], [], []
    for i, seed in enumerate(seeds):
        # get metrics
        imgs_pred = imgs_pred_all_seeds[:, i]  # (bs,3,h,w)
        all_imgs_pred.append(imgs_pred)
        # mse_b = mse_batch(imgs_pred * masks, imgs_gt * masks)
        ## better mse computation
        bs = len(imgs_pred)
        mse_b = (((imgs_gt*masks) - (imgs_pred*masks))**2 ).view(bs,-1).sum(dim=1) / masks.view(bs,-1).sum(dim=1)

        psnr_b = mse_to_psnr(mse_b)
        ssim_b = ssim_fn_batch(imgs_pred * masks, imgs_gt * masks)
        if do_lpips:
            lpips_b = lpips_fn_batch(imgs_pred * masks, imgs_gt * masks)
        else:
            lpips_b = torch.zeros_like(ssim_b)

        mse_train.append(mse_b[is_cam_idx_train])
        mse_test.append(mse_b[~is_cam_idx_train])

        psnr_train.append(psnr_b[is_cam_idx_train])
        psnr_test.append(psnr_b[~is_cam_idx_train])

        ssim_train.append(ssim_b[is_cam_idx_train])
        ssim_test.append(ssim_b[~is_cam_idx_train])

        lpips_train.append(lpips_b[is_cam_idx_train])
        lpips_test.append(lpips_b[~is_cam_idx_train])

        # compute residual and scale to (-1,1)
        residual = ((imgs_pred - imgs_gt) + 1) / 2

        # make grid
        nrow = len(imgs_gt)
        all_imgs_gt.append(imgs_gt_plot)
        grid_gt_plot = make_grid(imgs_gt_plot, nrow=nrow)
        grid_pred = make_grid(imgs_pred, nrow=nrow)
        grid_pred_masked = make_grid(imgs_pred * masks, nrow=nrow)
        grid_residual = make_grid(residual, nrow=nrow)
        grid = torch.cat(
            (grid_gt_plot, grid_pred, grid_pred_masked, grid_residual), dim=1)
        grid = grid.permute(1, 2, 0)  # (h,w,3)

        # get mean metrics for this seed only
        _mse_train_b = mse_b[is_cam_idx_train].mean().item()
        _mse_test_b = mse_b[~is_cam_idx_train].mean().item()
        _psnr_train_b = psnr_b[is_cam_idx_train].mean().item()
        _psnr_test_b = psnr_b[~is_cam_idx_train].mean().item()
        _ssim_train_b = ssim_b[is_cam_idx_train].mean().item()
        _ssim_test_b = ssim_b[~is_cam_idx_train].mean().item()
        _lpips_train_b = lpips_b[is_cam_idx_train].mean().item()
        _lpips_test_b = lpips_b[~is_cam_idx_train].mean().item()

        # get title and ticklabel information
        title = title_prefix + f" PSNR: train {_psnr_train_b:.3f}   test {_psnr_test_b:.3f}  |  "
        title += f"MSE: train {_mse_train_b:.3f}   test {_mse_test_b:.3f}  |  "
        title += f"SSIM: train {_ssim_train_b:.3f}   test {_ssim_test_b:.3f}  |  "
        title += f"LPIPS: train {_lpips_train_b:.3f}   test {_lpips_test_b:.3f}  |  "
        xticklabels = []
        for i, (is_train, p, m, s, l) in enumerate(
                zip(is_cam_idx_train, psnr_b, mse_b, ssim_b, lpips_b)):
            label = f"{p:.1f}\n{m:.4f}\n{s:.3f}\n{l:.3f}"
            if i == 0:
                label = "\n".join([
                    metric_label + metric for (metric_label, metric) in zip(
                        ['psnr ', 'mse ', 'ssim ', 'lpips'], label.split("\n"))
                ])
            if is_train:
                label += "\nTRAIN"
            xticklabels.append(label)
        img_ydim = imgs_gt.shape[2]
        xticks = np.linspace(0, grid.shape[1] - img_ydim,
                             len(xticklabels)) + img_ydim // 2

        f, axs = plt.subplots(figsize=(nrow, 5))
        axs.imshow(grid)
        axs.set_xticks(xticks)
        axs.set_xticklabels(xticklabels, fontsize=6)
        axs.set_yticks([])
        axs.set(title=title)
        figures.append(f)
        grids.append(grid)

    # summary metrics
    mse_train, mse_test = torch.cat(mse_train), torch.cat(mse_test)
    mse_train_mean, mse_train_std = mse_train.mean(), mse_train.std()
    mse_test_mean, mse_test_std = mse_test.mean(), mse_test.std()

    psnr_train, psnr_test = torch.cat(psnr_train), torch.cat(psnr_test)
    psnr_train_mean, psnr_train_std = psnr_train.mean(), psnr_train.std()
    psnr_test_mean, psnr_test_std = psnr_test.mean(), psnr_test.std()

    ssim_train, ssim_test = torch.cat(ssim_train), torch.cat(ssim_test)
    ssim_train_mean, ssim_train_std = ssim_train.mean(), ssim_train.std()
    ssim_test_mean, ssim_test_std = ssim_test.mean(), ssim_test.std()

    lpips_train, lpips_test = torch.cat(lpips_train), torch.cat(lpips_test)
    lpips_train_mean, lpips_train_std = lpips_train.mean(), lpips_train.std()
    lpips_test_mean, lpips_test_std = lpips_test.mean(), lpips_test.std()

    return dict(
        figures=figures,
        grids=grids,
        imgs_pred=all_imgs_pred,
        imgs_gt=imgs_gt,
        imgs_gt_plot=imgs_gt_plot,
        masks=masks,
        mse_train_mean=mse_train_mean.item(),
        mse_test_mean=mse_test_mean.item(),
        psnr_train_mean=psnr_train_mean.item(),
        psnr_test_mean=psnr_test_mean.item(),
        ssim_train_mean=ssim_train_mean.item(),
        ssim_test_mean=ssim_test_mean.item(),
        lpips_train_mean=lpips_train_mean.item(),
        lpips_test_mean=lpips_test_mean.item(),
    )

def mse_to_psnr(mse):
    """
    Compute PSNR given an MSE (we assume the maximum pixel value is 1).
    copied from FreeNerf. 
    Works for floats or tensors.
    """
    return -10. / np.log(10.) * np.log(mse)


def ssim_fn(x, y):
    """ skimage function does one image at a time """
    assert x.ndim == 3, x.shape[0] == 3
    return structural_similarity(x, y, channel_axis=0, data_range=1.0)


def ssim_fn_batch(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return torch.tensor([ssim_fn(x_, y_) for (x_, y_) in zip(x, y)])


def mse_batch(imgs_gt: torch.Tensor,
              imgs_pred: torch.Tensor,
              masks=None) -> torch.Tensor:
    """ For imgs (bs, ...), get mean mse for each batch element -> (bs,)"""
    bs = len(imgs_gt)
    mse = (imgs_gt - imgs_pred)**2
    mse_batch = mse.view(bs, -1).mean(1)
    return mse_batch


def lpips_fn_batch(imgs_gt, imgs_pred, lpips_fn=None):
    if lpips_fn is None:
        lpips_fn = LPIPS(net="vgg").cuda()
    # lpips needs to be in [-1,1]. Check that it's in (0,1) and then map it
    assert imgs_gt.min() >= 0 and imgs_gt.max() <= 1
    # imgs_gt = (imgs_gt * 2 - 1)
    # imgs_pred = (imgs_pred * 2 - 1)
    imgs_pred, imgs_gt = imgs_pred.cuda(), imgs_gt.cuda()
    imgs_pred = imgs_pred*2-1
    imgs_gt = imgs_gt*2-1
    #ipdb.set_trace()

    with torch.no_grad():
        res = lpips_fn(imgs_pred, imgs_gt)[:, 0, 0, 0].cpu()
    return res 


if __name__ == "__main__":
    pass
