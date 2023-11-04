"""
"""
import ipdb
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPTokenizer

sys.path.append(".")
sys.path.append("..")

import constants
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from models.xti_attention_processor import XTIAttenProc
from checkpoint_handler import CheckpointHandler
from utils import vis_utils
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from training.coach import Coach
from training.validate import ValidationHandler
from training.config import RunConfig
from training.dataset import TextualInversionDataset
from accelerate import Accelerator

@dataclass
class InferenceConfig:
    # Specifies which checkpoint iteration we want to load
    iteration: Optional[int] = None
    # The input directory containing the saved models and embeddings
    input_dir: Optional[Path] = None
    # Where the save the inference results to
    inference_dir: Optional[Path] = None
    # List of random seeds to run on
    seeds: List[int] = field(default_factory=lambda: [42])
    # Whether to run with torch.float16 or torch.float32
    torch_dtype: str = "fp16"
    # number of steps in diffusion model denoising
    num_denoising_steps: int = 30

    def __post_init__(self):
        if self.inference_dir is None:
            self.inference_dir = self.input_dir / "inference"
        self.inference_dir.mkdir(exist_ok=True, parents=True)
        self.torch_dtype = torch.float16 if self.torch_dtype == "fp16" else torch.float32

@pyrallis.wrap()
def main(infer_cfg: InferenceConfig):
    """ """
    # get the training config 
    mapper_checkpoint_path = infer_cfg.input_dir / f"mapper-steps-{infer_cfg.iteration}_view.pt"
    mapper_ckpt = torch.load(mapper_checkpoint_path)
    train_cfg_dict = CheckpointHandler.clean_config_dict(mapper_ckpt['cfg'])
    train_cfg = pyrallis.decode(RunConfig, train_cfg_dict) 
    train_cfg.eval.num_denoising_steps = infer_cfg.num_denoising_steps

    if train_cfg.data.camera_representation != 'dtu-12d':
        raise NotImplementedError("inference.py script only implemented for dtu dataset")

    # generate view tokens for DTU, and read the object token 
    lookup_camidx_to_view_token, _ = TextualInversionDataset.dtu_generate_dset_cam_tokens_params()
    cam_idxs = constants.DTU_SPLIT_IDXS['train'] + constants.DTU_SPLIT_IDXS['test']
    lookup_camidx_to_view_token = { k:v for (k,v) in lookup_camidx_to_view_token.items() if k in cam_idxs}
    placeholder_view_tokens = list(lookup_camidx_to_view_token.values()) # is all 64 view tokens
    placeholder_object_tokens = [train_cfg.data.placeholder_object_token]
    placeholder_tokens = placeholder_view_tokens + placeholder_object_tokens

    # get the model components 
    text_encoder, tokenizer, vae, unet, noise_scheduler = load_sd_model_components(train_cfg)

    # add the placholder tokens to the text encoder and tokenizer, and also get ther token_ids
    token_embeds, placeholder_token_ids, placeholder_view_token_ids, placeholder_object_token_ids = Coach._add_concept_token_to_tokenizer_static(
            train_cfg, placeholder_view_tokens, placeholder_object_tokens, tokenizer, text_encoder
        ) 

    # use the code from the validation handler to run DTU inference
    accelerator = Accelerator()
    validator = ValidationHandler(train_cfg, placeholder_view_tokens, 
        placeholder_view_token_ids, placeholder_object_tokens, 
        placeholder_object_token_ids, fixed_object_token=train_cfg.data.fixed_object_token_or_path, 
        weights_dtype=torch.float16)
    # validator.cfg.debug=1
    pipeline, results = validator.infer_dtu(accelerator, tokenizer, text_encoder, unet, vae,
                  num_images_per_prompt=None, seeds=infer_cfg.seeds, 
                  step=infer_cfg.iteration, prompts=None, return_instead_of_save=True)

    # the `results_all` is a dict with keys 'figures', 'grids', 'all_imgs_pred','all_imgs_gt', 'mse_train_mean', 'mse_test_mean', 'psnr_train_mean', 'psnr_test_mean', 'ssim_train_mean', 'ssim_test_mean', 'lpips_train_mean', 'lpips_test_mean'
    # this  can be loaded to access the original images
    # save results to image files in the config option `inference_dir`. If it wasn't 
    for i in range(len(results['figures'])):
        results['figures'][i].savefig(infer_cfg.inference_dir / f"preds_iter_{infer_cfg.iteration}_seed{infer_cfg.seeds[i]}.png", dpi=300)

    # first delete the figures and grids, since they're big and can be easily 
    # reconstructed from other things 
    for k in ["figures","grids","imgs_gt_plot"]:
        results.pop(k)
    torch.save(results, infer_cfg.inference_dir / f"results_all_iter_{infer_cfg.iteration}.pt")


def load_sd_model_components(train_cfg):
    """ 
    Load the SD model components separately AND add the placholder object and 
    view tokens to the tokenizer and text encoder 
    """
    # generate the tokenizer, and add the token_ids for the view and object tokens 
    text_encoder = NeTICLIPTextModel.from_pretrained(
            train_cfg.model.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=train_cfg.model.revision,
        )
    tokenizer = CLIPTokenizer.from_pretrained(
            train_cfg.model.pretrained_model_name_or_path,
            subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(
            train_cfg.model.pretrained_model_name_or_path,
            subfolder="vae",
            revision=train_cfg.model.revision)
    unet = UNet2DConditionModel.from_pretrained(
            train_cfg.model.pretrained_model_name_or_path,
            subfolder="unet",
            revision=train_cfg.model.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(
            train_cfg.model.pretrained_model_name_or_path,
            subfolder="scheduler")

    return text_encoder, tokenizer, vae, unet, noise_scheduler

if __name__ == '__main__':
    main()


