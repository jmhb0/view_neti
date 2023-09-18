import ipdb
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from transformers import CLIPTextConfig

from models.neti_mapper import NeTIMapper
from utils.types import NeTIBatch, MapperOutput


class NeTICLIPTextEmbeddings(nn.Module):
    """ Modification of CLIPTextEmbedding to allow for the use of a NeTIMapper to overwrite the concept token. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings,
                                               embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)))

    def set_mapper(self, mapper_object_lookup: Dict[str, NeTIMapper],
                   mapper_view: NeTIMapper, device='cuda'):
        self.mapper_object_lookup = mapper_object_lookup
        self.mapper_view = mapper_view

        # because its a dict, it won't put the mappers on cuda automatically
        for k, v in self.mapper_object_lookup.items():
            self.mapper_object_lookup[k] = v.to(device)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch: Optional[NeTIBatch] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if batch is not None:
            input_ids = batch.input_ids

        seq_length = input_ids.shape[
            -1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        ####################################################################
        # NeTI logic - Use mapper to overwrite the learnable token embedding
        ####################################################################
        bypass_outputs_object, bypass_outputs_view = None, None
        bypass_unconstrained_object, bypass_unconstrained_view = False, False
        output_bypass_alpha_object, output_bypass_alpha_view = None, None

        if batch is not None:
            ## for the object-mapper and view-mapper, separately look up the new
            # embedding values, get the output_bypass, and remap the embeddings

            if self.mapper_object_lookup is not None:
                # lookup the object idx
                assert torch.all(batch.input_ids_placeholder_object ==
                                 batch.input_ids_placeholder_object[0])
                idx = batch.input_ids_placeholder_object[0].item()
                mapper_object = self.mapper_object_lookup[idx]

                # compute the (t,l)-conditioned embedding
                mapper_object_outputs = mapper_object(
                    timestep=batch.timesteps.float(),
                    unet_layer=batch.unet_layers.float(),
                    input_ids_placeholder_view=None,
                    truncation_idx=batch.truncation_idx,
                )
                # strength of the output bypass -> to pass up to the encoder
                output_bypass_alpha_object = mapper_object_outputs.output_bypass_alpha

                # flag for whether we have the 'bypass_unconstrained' training mode
                bypass_unconstrained_object = mapper_object_outputs.bypass_unconstrained

                # word embedding vector
                word_embedding = mapper_object_outputs.word_embedding.to(
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # output_bypass if that flag is on
                if mapper_object.output_bypass:
                    bypass_outputs_object = mapper_object_outputs.bypass_output.to(
                        dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # replace special token embedding.
                locs = (input_ids ==
                        batch.input_ids_placeholder_object.unsqueeze(1))
                assert all(locs.sum(1) == 1)
                inputs_embeds[torch.where(locs)] = word_embedding

            # The second term in the if statement checks that input_view_ids are pasesd
            # This is bc, even if a mapper_view exists, we may still test prompts that
            # don't include that token.
            if self.mapper_view is not None and not all(
                    batch.input_ids_placeholder_view == -1):
                mapper_view_outputs = self.mapper_view(
                    timestep=batch.timesteps.float(),
                    unet_layer=batch.unet_layers.float(),
                    input_ids_placeholder_view=batch.
                    input_ids_placeholder_view,
                    truncation_idx=batch.truncation_idx)
                # strength of the output bypass -> to pass up to the encoder
                output_bypass_alpha_view = mapper_view_outputs.output_bypass_alpha

                # flag for whether we have the 'bypass_unconstrained' training mode
                bypass_unconstrained_view = mapper_view_outputs.bypass_unconstrained

                # word embedding vector
                word_embedding = mapper_view_outputs.word_embedding.to(
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # pull out the output_bypass if that flag is on
                if self.mapper_view.output_bypass:
                    bypass_outputs_view = mapper_view_outputs.bypass_output.to(
                        dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # replace special token embedding.
                locs = (
                    input_ids == batch.input_ids_placeholder_view.unsqueeze(1))
                assert all(locs.sum(1) == 1)
                inputs_embeds[torch.where(locs)] = word_embedding

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return (embeddings, bypass_outputs_object, bypass_outputs_view,
                bypass_unconstrained_object, bypass_unconstrained_view,
                output_bypass_alpha_object, output_bypass_alpha_view)
