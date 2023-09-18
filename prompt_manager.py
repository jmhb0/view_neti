import ipdb
from typing import Optional, List, Dict, Any

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer

import constants
from models.neti_clip_text_encoder import NeTICLIPTextModel
from utils.types import NeTIBatch


class PromptManager:
    """ 
    Class for computing all time and space embeddings for a given prompt. 
    
    MODIFIED: Before, would call `embed_prompt` passing a `text` that is a template
    like "A photo of a {}". The function would handle inserting of the 
    `self.placeholder_token`.
    Now, call `embed_prompt` with the already-filled string, like:
        "<view_0> a photo of a <car>". 
    Then the `embed_prompt` figures out which tokens are special tokens
    from `self.placeholder_view_token_ids` and `self.placeholder_object_token_ids`
    (Actually, it just passes those options to the text_encoder which handles it)
    """

    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 text_encoder: NeTICLIPTextModel,
                 timesteps: List[int] = constants.SD_INFERENCE_TIMESTEPS,
                 unet_layers: List[str] = constants.UNET_LAYERS,
                 placeholder_view_token_ids: List[int] = None,
                 placeholder_object_token_ids: List[int] = None,
                 torch_dtype: torch.dtype = torch.float32):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.timesteps = timesteps
        self.unet_layers = unet_layers
        self.placeholder_view_token_ids = placeholder_view_token_ids
        self.placeholder_object_token_ids = placeholder_object_token_ids
        self.dtype = torch_dtype

    def embed_prompt(self,
                     text: str,
                     truncation_idx: Optional[int] = None,
                     num_images_per_prompt: int = 1) -> List[Dict[str, Any]]:
        """
        Compute the conditioning vectors for the given prompt. We assume that the prompt is defined using `{}`
        
        for indicating where to place the placeholder token string. See constants.VALIDATION_PROMPTS for examples.
        """
        ids = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        ## MODIFIED: Read `ids` to extract which of the `placeholder_view_token_ids` is present.
        # (in training, this list is provided by the dataset)
        def get_input_ids_placeholder(ids, placeholder_token_ids):
            locs = torch.isin(ids, torch.tensor(placeholder_token_ids))
            if locs.sum()==0:
                return torch.tensor([-1])
            assert locs.sum(1), f"should be exactly 1 placeholder_view_token per prompt, for prompt [`{text}`]"
            input_ids_placeholder = ids[torch.where(locs)]
            return input_ids_placeholder

        input_ids_placeholder_object = get_input_ids_placeholder(ids, self.placeholder_object_token_ids)
        input_ids_placeholder_view = get_input_ids_placeholder(ids, self.placeholder_view_token_ids)

        # Compute embeddings for each timestep and each U-Net layer
        print(
            f"Computing embeddings over {len(self.timesteps)} timesteps and {len(self.unet_layers)} U-Net layers."
        )
        hidden_states_per_timestep = []
        device = self.text_encoder.device
        for timestep in tqdm(self.timesteps):
            _hs = {"this_idx": 0}.copy()
            for layer_idx, unet_layer in enumerate(self.unet_layers):
                batch = NeTIBatch(
                    input_ids=ids.to(device),
                    timesteps=timestep.unsqueeze(0).to(device),
                    unet_layers=torch.tensor(
                        layer_idx,
                        device=device).unsqueeze(0),
                    input_ids_placeholder_view=input_ids_placeholder_view.to(device),
                    input_ids_placeholder_object=input_ids_placeholder_object.to(device),
                    truncation_idx=truncation_idx)
                # self.tokenizer.TEST = True
                layer_hs, layer_hs_bypass = self.text_encoder(batch=batch)
                layer_hs = layer_hs[0].to(dtype=self.dtype)
                _hs[f"CONTEXT_TENSOR_{layer_idx}"] = layer_hs.repeat(
                    num_images_per_prompt, 1, 1)
                if layer_hs_bypass is not None:
                    layer_hs_bypass = layer_hs_bypass[0].to(dtype=self.dtype)
                    _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = layer_hs_bypass.repeat(
                        num_images_per_prompt, 1, 1)
            hidden_states_per_timestep.append(_hs)
        print("Done.")
        return hidden_states_per_timestep
