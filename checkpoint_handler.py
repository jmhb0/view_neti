import ipdb
import os
from pathlib import Path
from typing import Tuple, List, Literal

import pyrallis
import torch
from accelerate import Accelerator
from torch import nn
from transformers import CLIPTokenizer

from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from models.positional_encoding import NeTIPositionalEncoding, BasicEncoder, FourierPositionalEncoding
from training.config import RunConfig


class CheckpointHandler:

    def __init__(self, cfg: RunConfig, placeholder_view_tokens: List[str],
                 placeholder_view_token_ids: List[int],
                 placeholder_object_tokens: List[str],
                 placeholder_object_token_ids: List[int], save_root: Path):
        self.cfg = cfg
        self.placeholder_view_tokens = placeholder_view_tokens
        self.placeholder_view_token_ids = placeholder_view_token_ids
        self.placeholder_object_tokens = placeholder_object_tokens
        self.placeholder_object_token_ids = placeholder_object_token_ids
        self.save_root = save_root
        # aggregate the tokens
        self.placeholder_tokens = self.placeholder_view_tokens + self.placeholder_object_tokens
        self.placeholder_token_ids = self.placeholder_view_token_ids + self.placeholder_object_token_ids

    def save_model(self, text_encoder: NeTICLIPTextModel,
                   accelerator: Accelerator, embeds_save_name: str,
                   mapper_save_name: str):
        self.save_learned_embeds(text_encoder, accelerator, embeds_save_name)
        self.save_mapper(text_encoder, mapper_save_name)

    def save_learned_embeds(self, text_encoder: NeTICLIPTextModel,
                            accelerator: Accelerator, save_name: str):
        """
        Save learned embeddings. This embedding isn't really learned, but we'll add it to the tokenizer at inference
        to take the place of our placeholder token.
        (this is a weird thing to do, but whatever)
        """
        learned_embeds = accelerator.unwrap_model(
            text_encoder).get_input_embeddings().weight[
                self.placeholder_token_ids]
        learned_embeds = learned_embeds.detach().cpu()
        learned_embeds_dict = {
            t: v
            for (t, v) in zip(self.placeholder_tokens, learned_embeds)
        }
        torch.save(learned_embeds_dict, self.save_root / save_name)

    def save_mapper(self, text_encoder: NeTICLIPTextModel, save_name: str):
        """ Save the mapper and config to be used at inference. """
        cfg_ = RunConfig(**self.cfg.__dict__.copy())

        mapper_object_lookup, mapper_view = text_encoder.text_model.embeddings.mapper_object_lookup, text_encoder.text_model.embeddings.mapper_view

        if mapper_object_lookup is not None:
            state_dict = {"cfg": pyrallis.encode(cfg_), 'mappers': {}}

            for k, mapper_object in mapper_object_lookup.items():
                state_dict['mappers'][k] = {
                    "state_dict":
                    mapper_object.state_dict(),
                    "encoder":
                    mapper_object.encoder,
                    "placeholder_object_token":
                    mapper_object.placeholder_object_token,
                }
            fname = os.path.join(
                self.save_root,
                Path(save_name).stem + "_object" + Path(save_name).suffix)
            torch.save(state_dict, fname)

        if mapper_view is not None:
            state_dict = {
                "cfg": pyrallis.encode(cfg_),
                'mappers': {
                    'dummy_key': {
                        "state_dict": mapper_view.state_dict(),
                        "encoder": mapper_view.encoder,
                        'placeholder_object_token': "dummy",
                    }
                }
            }
            if hasattr(mapper_view, "encode_phi"):
                state_dict.update({"encode_phi": mapper_view.encode_phi})

            fname = os.path.join(
                self.save_root,
                Path(save_name).stem + "_view" + Path(save_name).suffix)
            torch.save(state_dict, fname)

    @staticmethod
    def load_mapper(
        mapper_path: Path,
        embedding_type: Literal["object", "view"] = "object",
        placeholder_view_tokens: List[str] = None,
        placeholder_view_token_ids: List[int] = None,
        placeholder_object_tokens: List[str] = None,
        placeholder_object_token_ids: List[int] = None,
    ) -> Tuple[RunConfig, NeTIMapper]:
        """ """
        mapper_ckpt = torch.load(mapper_path, map_location="cpu")
        # hacks
        if 'placeholder_view_tokens' in mapper_ckpt['cfg']['data'].keys():
            del mapper_ckpt['cfg']['data']['placeholder_view_tokens']

        for k in [
                'target_norm_object', 'target_norm_view',
                'pretrained_view_mapper', 'pretrained_view_mapper_key'
        ]:  
            if k in mapper_ckpt['cfg']['model'].keys():
                if mapper_ckpt['cfg']['model'][k] is None:
                    del mapper_ckpt['cfg']['model'][k]

        for k in ["validation_view_tokens", "eval_placeholder_object_tokens"]:
            if k in mapper_ckpt['cfg']['eval'].keys():
                if mapper_ckpt['cfg']['eval'][k] is None:
                    del mapper_ckpt['cfg']['eval'][k]

        for k in ['placeholder_object_tokens', 'train_data_subsets']:
            if k in mapper_ckpt['cfg']['data'].keys():
                if mapper_ckpt['cfg']['data'][k] is None:
                    del mapper_ckpt['cfg']['data'][k]

        cfg = pyrallis.decode(RunConfig, mapper_ckpt['cfg'])

        # handle the special case of getting the view token_ids from the tokenizer
        if embedding_type == "view":
            output_bypass = cfg.model.output_bypass_view
            assert placeholder_view_tokens is not None and placeholder_view_token_ids is not None
            target_norm = cfg.model.target_norm_view
        else:
            output_bypass = cfg.model.output_bypass_object
            placeholder_view_tokens, placeholder_view_token_ids = None, None
            target_norm = cfg.model.target_norm_object
            if target_norm is None and cfg.model.normalize_object_mapper_output:
                raise ValueError(
                    "need a target norm to pass to pretrained object mapper")

        # load this option that was added later
        bypass_unconstrained = False
        if 'bypass_unconstrained_object' in mapper_ckpt['cfg']['model'].keys():
            if embedding_type == "object":
                bypass_unconstrained = mapper_ckpt['cfg']['model'][
                    'bypass_unconstrained_object']
                output_bypass_alpha = mapper_ckpt['cfg']['model'].get(
                    'output_bypass_alpha_object', 0.2)
            else:
                bypass_unconstrained = mapper_ckpt['cfg']['model'][
                    'bypass_unconstrained_view']
                output_bypass_alpha = mapper_ckpt['cfg']['model'].get(
                    'output_bypass_alpha_object', 0.2)

        # Save to dict. Objects must be in this format because we can have
        # multiple object-mappers.
        neti_mapper_lookup = {}
        for k in mapper_ckpt['mappers'].keys():
            state_dict = mapper_ckpt['mappers'][k]['state_dict']
            encoder = mapper_ckpt['mappers'][k]['encoder']
            token = mapper_ckpt['mappers'][k]['placeholder_object_token']

            if embedding_type == "view":
                placeholder_object_token_id = "dummy"  # will be ignored anyway

            else:
                lookup_token_to_token_id = dict(
                    zip(placeholder_object_tokens,
                        placeholder_object_token_ids))
                placeholder_object_token_id = lookup_token_to_token_id[token]

            neti_mapper = NeTIMapper(
                embedding_type=embedding_type,
                placeholder_view_tokens=placeholder_view_tokens,
                placeholder_view_token_ids=placeholder_view_token_ids,
                output_dim=cfg.model.word_embedding_dim,
                arch_mlp_hidden_dims=cfg.model.arch_mlp_hidden_dims,
                use_nested_dropout=cfg.model.use_nested_dropout,
                nested_dropout_prob=cfg.model.nested_dropout_prob,
                norm_scale=target_norm,
                use_positional_encoding=cfg.model.
                use_positional_encoding_object,
                num_pe_time_anchors=cfg.model.num_pe_time_anchors,
                pe_sigmas=cfg.model.pe_sigmas,
                arch_view_net=cfg.model.arch_view_net,
                arch_view_mix_streams=cfg.model.arch_view_mix_streams,
                arch_view_disable_tl=cfg.model.arch_view_disable_tl,
                original_ti=cfg.model.original_ti,
                output_bypass=output_bypass,
                output_bypass_alpha=output_bypass_alpha,
                placeholder_object_token=token,
                bypass_unconstrained=bypass_unconstrained)

            neti_mapper.load_state_dict(state_dict, strict=True)

            # note that the encoder is only used in arch_view <= 14
            if isinstance(encoder, NeTIPositionalEncoding):
                encoder.w = nn.Parameter(mapper_ckpt['encoder'].w.cuda())
                neti_mapper.encoder = encoder.cuda()
            elif isinstance(encoder, BasicEncoder):
                encoder.normalized_timesteps = mapper_ckpt[
                    'encoder'].normalized_timesteps.cuda()
                encoder.normalized_unet_layers = mapper_ckpt[
                    'encoder'].normalized_unet_layers.cuda()
                neti_mapper.encoder = encoder.cuda()
            neti_mapper.cuda()
            neti_mapper.eval()
            neti_mapper_lookup[placeholder_object_token_id] = neti_mapper

        # if view, then return the only mapper; if object, return the dict of objects.
        mapper_out = neti_mapper_lookup[
            'dummy'] if embedding_type == "view" else neti_mapper_lookup

        return cfg, mapper_out

    @staticmethod
    def load_learned_embed_in_clip(
            learned_embeds_path: Path, text_encoder: NeTICLIPTextModel,
            tokenizer: CLIPTokenizer) -> Tuple[List[str], List[int]]:
        loaded_learned_embeds = torch.load(learned_embeds_path,
                                           map_location="cpu")

        # separate token and the embeds
        trained_tokens = list(loaded_learned_embeds.keys())
        embeds = list(loaded_learned_embeds.values())

        # cast to dtype of text_encoder
        dtype = text_encoder.get_input_embeddings().weight.dtype
        embeds = [e.to(dtype) for e in embeds]

        # add the tokens in tokenizer
        num_added_tokens = tokenizer.add_tokens(trained_tokens)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {trained_tokens[0]}. "
                f"Please pass a different `token` that is not already in the tokenizer."
            )

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        placeholder_token_ids = [
            tokenizer.convert_tokens_to_ids(t) for t in trained_tokens
        ]

        for idx, (token, token_id, embed) in enumerate(
                zip(trained_tokens, placeholder_token_ids, embeds)):
            text_encoder.get_input_embeddings().weight.data[token_id] = embed

        return trained_tokens, placeholder_token_ids
