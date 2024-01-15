import ipdb
from typing import Optional, Dict, Tuple, List, Union

import diffusers
import itertools
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from pathlib import Path
import pyrallis
from PIL import Image

from checkpoint_handler import CheckpointHandler
from constants import UNET_LAYERS
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from models.xti_attention_processor import XTIAttenProc
from training.config import RunConfig
from training.dataset import TextualInversionDataset
from training.logger import CoachLogger
from training.validate import ValidationHandler
from utils.types import NeTIBatch
from utils.utils import parameters_checksum
from utils import vis_utils
import torchvision.transforms as T


class Coach:

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.logger = CoachLogger(cfg=cfg)

        # Initialize some basic stuff
        self.accelerator = self._init_accelerator()
        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        if self.cfg.optim.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Initialize base models (but not neti-mappers yet)
        self.tokenizer, self.noise_scheduler, self.text_encoder, self.vae, self.unet = self._init_sd_models(
        )

        # Initialize dataset and dataloader
        self.train_dataset = self._init_dataset()
        self.train_dataloader = self._init_dataloader(
            dataset=self.train_dataset)
        self.placeholder_object_tokens = self.train_dataset.placeholder_object_tokens
        self.placeholder_view_tokens = self.train_dataset.placeholder_view_tokens
        self.placeholder_tokens = self.train_dataset.placeholder_tokens
        self.fixed_object_token = self.train_dataset.fixed_object_token
        if self.cfg.eval.validation_view_tokens is not None:
            assert all([
                v in self.placeholder_view_tokens
                for v in self.cfg.eval.validation_view_tokens
            ])

        if self.cfg.log.save_dataset_images:
            self.save_dataset_images()

        # add novel concepts
        self.load_pretrained_object_neti = True if self.cfg.data.fixed_object_token_or_path is not None and Path(
            self.cfg.data.fixed_object_token_or_path
        ).suffix == '.pt' else False
        # self.token_embeds, self.placeholder_token_ids, self.placeholder_view_token_ids, self.placeholder_object_token_ids = self._add_concept_token_to_tokenizer(
        # )
        # alternative way to do it
        self.token_embeds, self.placeholder_token_ids, self.placeholder_view_token_ids, self.placeholder_object_token_ids = Coach._add_concept_token_to_tokenizer_static(
            self.cfg, self.train_dataset.placeholder_view_tokens, self.train_dataset.placeholder_object_tokens, self.tokenizer, self.text_encoder
        )

        self.cfg.data.placeholder_view_tokens = self.placeholder_view_tokens

        # Initilize neti mapping objects, and finish preparing all the models
        neti_mapper_object_lookup, neti_mapper_view, self.loaded_iteration = self._init_neti_mapper(
        )
        self.text_encoder.text_model.embeddings.set_mapper(
            neti_mapper_object_lookup, neti_mapper_view)
        self._freeze_all_modules()
        self._set_attn_processor()

        # Initialize optimizer and scheduler

        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_scheduler(optimizer=self.optimizer)

        # Prepare everything with accelerator
        self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.text_encoder, self.optimizer, self.train_dataloader,
            self.lr_scheduler)

        # Reconfigure some parameters that we'll need for training
        self.weight_dtype = self._get_weight_dtype()
        self._set_model_weight_dtypes(weight_dtype=self.weight_dtype)
        self._init_trackers()

        self.validator = ValidationHandler(
            cfg=self.cfg,
            placeholder_view_tokens=self.placeholder_view_tokens,
            placeholder_view_token_ids=self.placeholder_view_token_ids,
            placeholder_object_tokens=self.placeholder_object_tokens,
            placeholder_object_token_ids=self.placeholder_object_token_ids,
            fixed_object_token=self.fixed_object_token,
            weights_dtype=self.weight_dtype)

        self.checkpoint_handler = CheckpointHandler(
            cfg=self.cfg,
            placeholder_view_tokens=self.placeholder_view_tokens,
            placeholder_view_token_ids=self.placeholder_view_token_ids,
            placeholder_object_tokens=self.placeholder_object_tokens,
            placeholder_object_token_ids=self.placeholder_object_token_ids,
            save_root=self.cfg.log.exp_dir)
    # ipdb.set_trace()

        #### tmp - for testing if the loaded object token works ####
        if 0:
            test_prompts = ["A photo of a <car>"]
            self.validator.infer(accelerator=self.accelerator,
                                 tokenizer=self.tokenizer,
                                 text_encoder=self.text_encoder,
                                 unet=self.unet,
                                 vae=self.vae,
                                 prompts=test_prompts,
                                 num_images_per_prompt=1,
                                 seeds=[42, 420, 501],
                                 step=0)

    def train(self):
        total_batch_size = self.cfg.optim.train_batch_size * self.accelerator.num_processes * \
                           self.cfg.optim.gradient_accumulation_steps
        self.logger.log_start_of_training(total_batch_size=total_batch_size,
                                          num_samples=len(self.train_dataset))

        global_step = self._set_global_step()
        progress_bar = tqdm(range(global_step, self.cfg.optim.max_train_steps),
                            disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        orig_embeds_params = self.accelerator.unwrap_model(
            self.text_encoder).get_input_embeddings().weight.data.clone()

        while global_step < self.cfg.optim.max_train_steps:

            self.text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):
                if self.cfg.learnable_mode == 3:
                    self.train_dataset.reset_sampled_object()

                with self.accelerator.accumulate(self.text_encoder):
                    ## following commented code is to check that weights are updating
                    # cksm_object = parameters_checksum(list(self.text_encoder.text_model.embeddings.mapper_object_lookup.values())[0])
                    # cksm_view = parameters_checksum(self.text_encoder.text_model.embeddings.mapper_view)
                    # print(f"checksums object {cksm_object} | view {cksm_view}")

                    # Convert images to latent space
                    latent_batch = batch["pixel_values"].to(
                        dtype=self.weight_dtype)
                    latents = self.vae.encode(
                        latent_batch).latent_dist.sample().detach()
                    latents = latents * self.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        low=0,
                        high=self.noise_scheduler.config.num_train_timesteps,
                        size=(bsz, ),
                        device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    _hs = self.get_text_conditioning(
                        input_ids=batch['input_ids'],
                        input_ids_placeholder_object=batch[
                            'input_ids_placeholder_object'],
                        input_ids_placeholder_view=batch[
                            'input_ids_placeholder_view'],
                        timesteps=timesteps,
                        original_ti=self.cfg.model.original_ti,
                        device=latents.device)

                    # Predict the noise residual
                    model_pred = self.unet(noisy_latents, timesteps,
                                           _hs).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(
                            latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )

                    loss = F.mse_loss(model_pred.float(),
                                      target.float(),
                                      reduction="mean")
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Let's make sure we don't update any embedding weights besides the newly added token
                    # This isn't really needed, but we'll keep it for consistency with the original code
                    index_no_updates = torch.isin(
                        torch.arange(len(self.tokenizer)),
                        torch.Tensor(self.placeholder_token_ids))
                    with torch.no_grad():
                        self.accelerator.unwrap_model(
                            self.text_encoder).get_input_embeddings(
                            ).weight[index_no_updates] = orig_embeds_params[
                                index_no_updates]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.logger.update_step(step=global_step)
                    if self._should_save(global_step=global_step):
                        self.checkpoint_handler.save_model(
                            text_encoder=self.text_encoder,
                            accelerator=self.accelerator,
                            embeds_save_name=
                            f"learned_embeds-steps-{global_step}.bin",
                            mapper_save_name=f"mapper-steps-{global_step}.pt")
                    if self._should_eval(global_step=global_step):
                        self.validator.infer(
                            accelerator=self.accelerator,
                            tokenizer=self.tokenizer,
                            text_encoder=self.text_encoder,
                            unet=self.unet,
                            vae=self.vae,
                            num_images_per_prompt=self.cfg.eval.
                            num_validation_images,
                            seeds=self.cfg.eval.validation_seeds,
                            step=global_step)

                logs = {
                    "total_loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.cfg.optim.max_train_steps:
                    break

        # Save the final model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.checkpoint_handler.save_model(
                text_encoder=self.text_encoder,
                accelerator=self.accelerator,
                embeds_save_name=f"learned_embeds-final.bin",
                mapper_save_name=f"mapper-final.pt")
        self.accelerator.end_training()

    def get_text_conditioning(
            self,
            input_ids: torch.Tensor,
            timesteps: torch.Tensor,
            input_ids_placeholder_object: List[int],
            input_ids_placeholder_view: List[int],
            device: torch.device,
            original_ti: bool = False) -> Union[Dict, torch.Tensor]:
        """ Compute the text conditioning for the current batch of images using our text encoder over-ride. 
        If original_ti, then just return the last layer directly
        """
        _hs = {"this_idx": 0}

        for layer_idx, unet_layer in enumerate(UNET_LAYERS):
            neti_batch = NeTIBatch(
                input_ids=input_ids,
                input_ids_placeholder_object=input_ids_placeholder_object,
                input_ids_placeholder_view=input_ids_placeholder_view,
                timesteps=timesteps,
                unet_layers=torch.tensor(layer_idx, device=device).repeat(
                    timesteps.shape[0]))
            layer_hidden_state, layer_hidden_state_bypass = self.text_encoder(
                batch=neti_batch)
            layer_hidden_state = layer_hidden_state[0].to(
                dtype=self.weight_dtype)
            _hs[f"CONTEXT_TENSOR_{layer_idx}"] = layer_hidden_state
            if layer_hidden_state_bypass is not None:
                layer_hidden_state_bypass = layer_hidden_state_bypass[0].to(
                    dtype=self.weight_dtype)
                _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = layer_hidden_state_bypass

            # if running original ti, only run this
            if original_ti:
                return layer_hidden_state

        return _hs

    def _set_global_step(self) -> int:
        global_step = 0
        if self.loaded_iteration is not None:
            global_step = self.loaded_iteration
        self.logger.update_step(step=global_step)
        return global_step

    @staticmethod
    def _add_concept_token_to_tokenizer_static(cfg, placeholder_view_tokens, 
        placeholder_object_tokens, tokenizer, text_encoder
        ):
        """ modifies the tokenizer and text_encoder in place """
        placeholder_tokens = placeholder_view_tokens + placeholder_object_tokens
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens == 0:
            raise ValueError(
                f"No new tokens were added to the tokenizer"
                f"Please pass a different `placeholder_token` that is not already in the tokenizer."
            )

        # extract all the placeholder ids
        placeholder_view_token_ids = tokenizer.convert_tokens_to_ids(
            placeholder_view_tokens)
        placeholder_object_token_ids = tokenizer.convert_tokens_to_ids(
            placeholder_object_tokens)
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(
            placeholder_tokens)
        assert set(placeholder_view_token_ids).union(
            set(placeholder_object_token_ids)) == set(placeholder_token_ids)


        ### TODO - this should handle a list of super_category tokens
        # Convert the super_category_token, placeholder_token to ids
        super_category_object_token_id = tokenizer.encode(
            cfg.data.super_category_object_token,
            add_special_tokens=False)
        super_category_view_token_id = tokenizer.encode(
            cfg.data.super_category_view_token, add_special_tokens=False)

        super_token_ids = super_category_object_token_id + super_category_view_token_id

        # Check if super_category_token is a single token or a sequence of tokens
        if len(super_category_object_token_id) != 1:
            raise ValueError(
                f"object supercategory [self.cfg.data.super_category_object_token] not in the vocabulary"
            )
        if len(super_category_view_token_id) != 1:
            raise ValueError(
                "view supercategory [self.cfg.data.super_category_view_token] not in the vocabulary"
            )
        super_category_object_token_id, super_category_view_token_id = super_category_object_token_id[
            0], super_category_view_token_id[0]

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialize the newly added placeholder token with the embeddings of the super category token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_view_token_ids] = token_embeds[
            super_category_view_token_id].clone().unsqueeze(0).repeat(
                len(placeholder_view_token_ids), 1)
        token_embeds[placeholder_object_token_ids] = token_embeds[
            super_category_object_token_id].clone().unsqueeze(0).repeat(
                len(placeholder_object_token_ids), 1)

        ## Compute the norm of the super category token embedding for scaling mapper output
        cfg.model.target_norm_view = None
        cfg.model.target_norm_object = None

        if cfg.model.normalize_view_mapper_output:
            if super_category_view_token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"super_category_view_token [{cfg.data.super_category_object_token}] ",
                    " is unknown to the tokenizer")
            cfg.model.target_norm_view = token_embeds[
                super_category_view_token_id].norm().item()
        if cfg.model.normalize_object_mapper_output:
            if super_category_object_token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"super_category_view_token [{super_category_object_token_id}] ",
                    " is unknown to the tokenizer")
            cfg.model.target_norm_object = token_embeds[
                super_category_object_token_id].norm().item()

        return token_embeds, placeholder_token_ids, placeholder_view_token_ids, placeholder_object_token_ids

    def _add_concept_token_to_tokenizer(self) -> Tuple[torch.Tensor, int]:
        """
        Adds the concept token to the tokenizer and initializes it with the embeddings of the super category token.
        The super category token will also be used for computing the norm for rescaling the mapper output.
        """

        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)
        if num_added_tokens == 0:
            raise ValueError(
                f"No new tokens were added to the tokenizer"
                f"Please pass a different `placeholder_token` that is not already in the tokenizer."
            )

        # extract all the placeholder ids
        placeholder_view_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.train_dataset.placeholder_view_tokens)
        placeholder_object_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.train_dataset.placeholder_object_tokens)
        placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.placeholder_tokens)
        assert set(placeholder_view_token_ids).union(
            set(placeholder_object_token_ids)) == set(placeholder_token_ids)


        ### TODO - this should handle a list of super_category tokens
        # Convert the super_category_token, placeholder_token to ids
        super_category_object_token_id = self.tokenizer.encode(
            self.cfg.data.super_category_object_token,
            add_special_tokens=False)
        super_category_view_token_id = self.tokenizer.encode(
            self.cfg.data.super_category_view_token, add_special_tokens=False)

        super_token_ids = super_category_object_token_id + super_category_view_token_id

        # Check if super_category_token is a single token or a sequence of tokens
        if len(super_category_object_token_id) != 1:
            raise ValueError(
                f"object supercategory [self.cfg.data.super_category_object_token] not in the vocabulary"
            )
        if len(super_category_view_token_id) != 1:
            raise ValueError(
                "view supercategory [self.cfg.data.super_category_view_token] not in the vocabulary"
            )
        super_category_object_token_id, super_category_view_token_id = super_category_object_token_id[
            0], super_category_view_token_id[0]

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialize the newly added placeholder token with the embeddings of the super category token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_view_token_ids] = token_embeds[
            super_category_view_token_id].clone().unsqueeze(0).repeat(
                len(placeholder_view_token_ids), 1)
        token_embeds[placeholder_object_token_ids] = token_embeds[
            super_category_object_token_id].clone().unsqueeze(0).repeat(
                len(placeholder_object_token_ids), 1)

        ## Compute the norm of the super category token embedding for scaling mapper output
        self.cfg.model.target_norm_view = None
        self.cfg.model.target_norm_object = None

        if self.cfg.model.normalize_view_mapper_output:
            if super_category_view_token_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    f"super_category_view_token [{self.cfg.data.super_category_object_token}] ",
                    " is unknown to the tokenizer")
            self.cfg.model.target_norm_view = token_embeds[
                super_category_view_token_id].norm().item()
        if self.cfg.model.normalize_object_mapper_output:
            if super_category_object_token_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    f"super_category_view_token [{super_category_object_token_id}] ",
                    " is unknown to the tokenizer")
            self.cfg.model.target_norm_object = token_embeds[
                super_category_object_token_id].norm().item()

        return token_embeds, placeholder_token_ids, placeholder_view_token_ids, placeholder_object_token_ids

    def save_dataset_images(self) -> None:
        n_max = 100
        fnames = self.train_dataset.image_paths_flattened
        if len(fnames) > n_max:
            fnames = fnames[:n_max]
            save_fname = self.cfg.log.exp_dir / 'dataset_first_100.png'
        else:
            save_fname = self.cfg.log.exp_dir / 'dataset.png'

        images = [Image.open(f) for f in fnames]
        grid = vis_utils.get_image_grid(images)
        grid = vis_utils.downsample_image(grid, 0.2)
        grid.save(save_fname)

    def _init_neti_mapper(self) -> Tuple[NeTIMapper, Optional[int]]:
        neti_mapper_object_lookup, neti_mapper_view = None, None

        if self.cfg.learnable_mode not in (0, 1, 2, 3, 4, 5):
            raise NotImplementedError()

        loaded_iteration = None
        # this loading func is from prior codebase.
        if self.cfg.model.mapper_checkpoint_path:
            raise NotImplementedError("Check this implementation is right")
            # This isn't 100% resuming training since we don't save the optimizer, but it's close enough
            _, neti_mapper = CheckpointHandler.load_mapper(
                self.cfg.model.mapper_checkpoint_path)
            loaded_iteration = int(
                self.cfg.model.mapper_checkpoint_path.stem.split("-")[-1])

        if self.cfg.learnable_mode in (0, 2, 3, 4, 5):

            # next line only matters if cfg.model.original_ti=1 ... is a hack
            original_ti_init_embed = self.text_encoder.get_input_embeddings(
            ).weight.data[self.tokenizer.convert_tokens_to_ids(
                self.cfg.data.placeholder_object_token)]

            # multiple object mappers if jointly training multiple objects
            super_category_object_tokens = self.cfg.data.super_category_object_tokens if self.cfg.learnable_mode == 3 else [
                self.cfg.data.super_category_object_token
            ]

            neti_mapper_object_lookup = {}
            for (placeholder_object_token, placeholder_object_token_id) in zip(
                    self.placeholder_object_tokens,
                    self.placeholder_object_token_ids):

                neti_mapper_object = NeTIMapper(
                    embedding_type="object",
                    placeholder_view_tokens=None,
                    placeholder_view_token_ids=None,
                    placeholder_object_token=placeholder_object_token,
                    output_dim=self.cfg.model.word_embedding_dim,
                    arch_mlp_hidden_dims=self.cfg.model.arch_mlp_hidden_dims,
                    use_nested_dropout=self.cfg.model.use_nested_dropout,
                    nested_dropout_prob=self.cfg.model.nested_dropout_prob,
                    norm_scale=self.cfg.model.target_norm_object,
                    use_positional_encoding=self.cfg.model.
                    use_positional_encoding_object,
                    num_pe_time_anchors=self.cfg.model.num_pe_time_anchors,
                    pe_sigmas=self.cfg.model.pe_sigmas,
                    arch_view_net=self.cfg.model.arch_view_net,
                    arch_view_mix_streams=self.cfg.model.arch_view_mix_streams,
                    arch_view_disable_tl=self.cfg.model.arch_view_disable_tl,
                    original_ti=self.cfg.model.original_ti,
                    original_ti_init_embed=original_ti_init_embed,
                    output_bypass=self.cfg.model.output_bypass_object,
                    output_bypass_alpha=self.cfg.model.
                    output_bypass_alpha_object,
                    bypass_unconstrained=self.cfg.model.
                    bypass_unconstrained_object)

                neti_mapper_object_lookup[
                    placeholder_object_token_id] = neti_mapper_object

        if self.cfg.learnable_mode in (1, 2, 3):
            if self.load_pretrained_object_neti:
                _, neti_mapper_object = CheckpointHandler.load_mapper(
                    self.cfg.data.fixed_object_token_or_path, "object",
                    self.tokenizer)

            # hack
            original_ti_init_embed = self.text_encoder.get_input_embeddings(
            ).weight.data[self.tokenizer.convert_tokens_to_ids("view")]

            neti_mapper_view = NeTIMapper(
                embedding_type="view",
                placeholder_view_tokens=self.placeholder_view_tokens,
                placeholder_view_token_ids=self.placeholder_view_token_ids,
                placeholder_object_token=None,
                output_dim=self.cfg.model.word_embedding_dim,
                arch_mlp_hidden_dims=self.cfg.model.arch_mlp_hidden_dims,
                use_nested_dropout=self.cfg.model.use_nested_dropout,
                nested_dropout_prob=self.cfg.model.nested_dropout_prob,
                norm_scale=self.cfg.model.target_norm_view,
                use_positional_encoding=self.cfg.model.
                use_positional_encoding_view,
                num_pe_time_anchors=self.cfg.model.num_pe_time_anchors,
                pe_sigmas=self.cfg.model.pe_sigmas,
                arch_view_net=self.cfg.model.arch_view_net,
                arch_view_mix_streams=self.cfg.model.arch_view_mix_streams,
                arch_view_disable_tl=self.cfg.model.arch_view_disable_tl,
                output_bypass=self.cfg.model.output_bypass_view,
                original_ti=self.cfg.model.original_ti,
                original_ti_init_embed=original_ti_init_embed,
                output_bypass_alpha=self.cfg.model.output_bypass_alpha_view,
                bypass_unconstrained=self.cfg.model.bypass_unconstrained_view)

        elif self.cfg.learnable_mode in (4, 5):
            # load pretrained view mapperi
            cfg_pretrained_view_mapper, neti_mapper_view = CheckpointHandler.load_mapper(
                self.cfg.model.pretrained_view_mapper,
                "view",
                placeholder_view_tokens=self.placeholder_view_tokens,
                placeholder_view_token_ids=self.placeholder_view_token_ids)
            # save the pretrained config
            fname_cfg_pretrain = self.cfg.log.exp_dir / "config_view_pretrained.yaml"
            with (fname_cfg_pretrain).open('w') as f:
                pyrallis.dump(cfg_pretrained_view_mapper, f)

        return neti_mapper_object_lookup, neti_mapper_view, loaded_iteration

    def _init_sd_models(self):
        tokenizer = self._init_tokenizer()
        noise_scheduler = self._init_noise_scheduler()
        text_encoder = self._init_text_encoder()
        vae = self._init_vae()
        unet = self._init_unet()
        return tokenizer, noise_scheduler, text_encoder, vae, unet

    def _init_tokenizer(self) -> CLIPTokenizer:
        tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            subfolder="tokenizer")
        return tokenizer

    def _init_noise_scheduler(self) -> DDPMScheduler:
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            subfolder="scheduler")
        return noise_scheduler

    def _init_text_encoder(self) -> NeTICLIPTextModel:
        text_encoder = NeTICLIPTextModel.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.cfg.model.revision,
        )
        return text_encoder

    def _init_vae(self) -> AutoencoderKL:
        vae = AutoencoderKL.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.cfg.model.revision)
        return vae

    def _init_unet(self) -> UNet2DConditionModel:
        unet = UNet2DConditionModel.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.cfg.model.revision)
        return unet

    def _freeze_all_modules(self):
        if self.cfg.learnable_mode not in (0, 1, 2, 3, 4, 5):
            raise NotImplementedError()

        # Freeze vae, unet, text model
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # Freeze all parameters except for the mapper in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(
            False)

        # Train the mapper
        def enable_mapper_training(mapper):
            mapper.requires_grad_(True)
            mapper.train()

        # train object mappers
        if self.cfg.learnable_mode in (0, 2, 3, 4, 5):
            for mapper in self.text_encoder.text_model.embeddings.mapper_object_lookup.values(
            ):
                enable_mapper_training(mapper)

        # train view mappers
        if self.cfg.learnable_mode in (1, 2, 3, 4):
            enable_mapper_training(
                self.text_encoder.text_model.embeddings.mapper_view)
            # todo: option to keep learning a pretrained object mapper

        if self.cfg.optim.gradient_checkpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            self.unet.train()
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

    def _set_attn_processor(self):
        self.unet.set_attn_processor(XTIAttenProc())

    def _init_dataset(self) -> TextualInversionDataset:
        dataset = TextualInversionDataset(
            learnable_mode=self.cfg.learnable_mode,
            fixed_object_token_or_path=self.cfg.data.
            fixed_object_token_or_path,
            data_root=self.cfg.data.train_data_dir,
            train_data_subsets=self.cfg.data.train_data_subsets,
            placeholder_object_tokens=self.cfg.data.placeholder_object_tokens,
            tokenizer=self.tokenizer,
            size=self.cfg.data.resolution,
            placeholder_object_token=self.cfg.data.placeholder_object_token,
            repeats=self.cfg.data.repeats,
            center_crop=self.cfg.data.center_crop,
            caption_strategy=self.cfg.data.caption_strategy,
            camera_representation=self.cfg.data.camera_representation,
            dtu_lighting=self.cfg.data.dtu_lighting,
            dtu_subset=self.cfg.data.dtu_subset,
            dtu_preprocess_key=self.cfg.data.dtu_preprocess_key,
            augmentation_key=self.cfg.data.augmentation_key,
            set="train")
        return dataset

    def _init_dataloader(self,
                         dataset: Dataset) -> torch.utils.data.DataLoader:

        def custom_collate_fn(batch, dset):
            # #
            # example = {}
            # n = len(batch)
            # for k in batch[0].keys():
            #     if type(batch[i][k]) is torch.Tensor:
            #         example[k] = torch.tensor([batch[i][k] for i in range(n)])
            raise NotImplementedError("batching")
            """ little hack to access the dataset object """
            dset.reset_sampled_object()
            return batch

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.optim.train_batch_size,
            shuffle=True,
            # collate_fn=lambda x: custom_collate_fn(x, dataset),
            num_workers=self.cfg.data.dataloader_num_workers)
        return dataloader

    def _init_optimizer(self) -> torch.optim.Optimizer:
        if self.cfg.optim.scale_lr:
            self.cfg.optim.learning_rate = (
                self.cfg.optim.learning_rate *
                self.cfg.optim.gradient_accumulation_steps *
                self.cfg.optim.train_batch_size *
                self.accelerator.num_processes)

        ## choose the optimizable params depending on the learning mode
        learnable_params = []

        # object - all modes except 1
        if self.cfg.learnable_mode in (0, 2, 3, 4, 5):
            for mapper in self.text_encoder.text_model.embeddings.mapper_object_lookup.values(
            ):
                learnable_params.append(mapper.parameters())

        # view - all modes except 0 and 5
        if self.cfg.learnable_mode in (1, 2, 3, 4):
            learnable_params.append(self.text_encoder.text_model.embeddings.
                                    mapper_view.parameters())

        learnable_params_ = itertools.chain.from_iterable(learnable_params)
        optimizer = torch.optim.AdamW(
            learnable_params_,  # neti-mappers only
            lr=self.cfg.optim.learning_rate,
            betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2),
            weight_decay=self.cfg.optim.adam_weight_decay,
            eps=self.cfg.optim.adam_epsilon,
        )
        return optimizer

    def _init_scheduler(
            self,
            optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        lr_scheduler = get_scheduler(
            self.cfg.optim.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.optim.lr_warmup_steps *
            self.cfg.optim.gradient_accumulation_steps,
            num_training_steps=self.cfg.optim.max_train_steps *
            self.cfg.optim.gradient_accumulation_steps,
        )
        return lr_scheduler

    def _init_accelerator(self) -> Accelerator:
        accelerator_project_config = ProjectConfiguration(
            total_limit=self.cfg.log.checkpoints_total_limit)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.optim.
            gradient_accumulation_steps,
            mixed_precision=self.cfg.optim.mixed_precision,
            log_with=self.cfg.log.report_to,
            logging_dir=self.cfg.log.logging_dir,
            project_config=accelerator_project_config,
        )
        self.logger.log_message(accelerator.state)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        return accelerator

    def _set_model_weight_dtypes(self, weight_dtype: torch.dtype):
        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)

    def _get_weight_dtype(self) -> torch.dtype:
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype

    def _init_trackers(self):
        config = pyrallis.encode(self.cfg)

        # tensorboard only accepts dicts with entries having (int, float, str, bool, or torch.Tensor)
        config_tensorboard = {**config['log'], **config['model']}
        if config_tensorboard is not None and 'pe_sigmas' in config_tensorboard.keys(
        ):
            del config_tensorboard['pe_sigmas']  #(doesn't like dictionaries)
            if 'pe_sigmas_view' in config_tensorboard.keys():
                del config_tensorboard[
                    'pe_sigmas_view']  #(doesn't like dictionaries)

        # give wandb the full logging dictionary bc it knows how to parse it.
        init_kwargs = {
            'wandb': {
                'config': config,
                'name': config['log']['exp_name'],
            },
        }

        # init trackers
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("view-neti",
                                           config=config_tensorboard,
                                           init_kwargs=init_kwargs)

    def _should_save(self, global_step: int) -> bool:
        return global_step % self.cfg.log.save_steps == 0

    def _should_eval(self, global_step: int) -> bool:
        return self.cfg.eval.validation_prompts is not None and global_step % self.cfg.eval.validation_steps == 0
