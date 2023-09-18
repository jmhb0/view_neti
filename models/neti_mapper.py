import ipdb
import random
from typing import Optional, List, Literal

import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch import nn
import numpy as np

from constants import UNET_LAYERS
from models.positional_encoding import NeTIPositionalEncoding, BasicEncoder, PositionalEncoding, FourierPositionalEncoding, FourierPositionalEncodingNDims
from models.mlp import MLP
from utils.types import PESigmas, MapperOutput
from utils.utils import num_to_string, string_to_num
from training.dataset import TextualInversionDataset


class NeTIMapper(nn.Module):
    """ Main logic of our NeTI mapper. """

    def __init__(
        self,
        embedding_type: Literal['object', 'view'],
        output_dim: int = 768,
        unet_layers: List[str] = UNET_LAYERS,
        arch_mlp_hidden_dims: int = 128,
        use_nested_dropout: bool = True,
        nested_dropout_prob: float = 0.5,
        norm_scale: Optional[torch.Tensor] = None,
        use_positional_encoding=1,
        num_pe_time_anchors: int = 10,
        pe_sigmas: PESigmas = PESigmas(sigma_t=0.03,
                                       sigma_l=2.0,
                                       sigma_phi=1.0),
        output_bypass: bool = True,
        placeholder_view_tokens: List[str] = None,
        placeholder_view_token_ids: torch.Tensor = None,
        arch_view_net: int = 0,
        arch_view_mix_streams: int = 0,
        arch_view_disable_tl: bool = True,
        original_ti_init_embed=None,
        original_ti: bool = False,
        bypass_unconstrained: bool = True,
        output_bypass_alpha: float = 0.2,
        placeholder_object_token: str = None,
    ):
        """
        Args:
        embedding_type: whether the Neti-mapper should learn object or view 
            control. View-control will condition on camera pose as well. MLP 
            architecture is also different. 
        placeholder_view_tokens: all possible view_tokens used for training. 
            Ignored if embedding_type=='object'.
        placeholder_view_tokens_ids: token ids for `placeholder_view_tokens`
        arch_view_disable_tl: do not condition on timestep and unet layer (t,l)
        original_ti: run the 'original TI'
        bypass_unconstrained: passed through in the output
        """
        super().__init__()
        self.embedding_type = embedding_type
        self.arch_view_net = arch_view_net
        self.use_nested_dropout = use_nested_dropout
        self.nested_dropout_prob = nested_dropout_prob
        self.arch_mlp_hidden_dims = arch_mlp_hidden_dims
        self.norm_scale = norm_scale
        self.original_ti = original_ti
        self.arch_view_disable_tl = arch_view_disable_tl
        self.original_ti_init_embed = original_ti_init_embed
        self.output_bypass_alpha = output_bypass_alpha
        self.num_unet_layers = len(unet_layers)
        self.placeholder_object_token = placeholder_object_token  # does nothing
        if original_ti and output_bypass:
            raise ValueError(
                f"If doing cfg.model.original_ti=[True]",
                f" then you cannot have cfg.model.original_ti=[True]")
        self.output_bypass = output_bypass
        if self.output_bypass:
            output_dim *= 2  # Output two vectors

        # for view mappers, prepare some class properties for the view tokens
        if self.embedding_type == "view":
            self.placeholder_view_tokens = placeholder_view_tokens
            self.placeholder_view_token_ids = placeholder_view_token_ids
            self._prepare_view_token_param_lookup(rescale_min_max=True)

        # set up positional encoding. For older experiments (arch_view_net<14),
        # use the legacy (t,l) conditioning. For later exps, call func for setup
        self.pe_sigmas = pe_sigmas
        if self.arch_view_net <= 14:
            self.use_positional_encoding = use_positional_encoding
            if type(self.use_positional_encoding) is bool:
                self.use_positional_encoding = int(
                    self.use_positional_encoding)
            if self.use_positional_encoding == 1:
                self.encoder = NeTIPositionalEncoding(
                    sigma_t=pe_sigmas.sigma_t,
                    sigma_l=pe_sigmas.sigma_l).cuda()
                self.input_dim = num_pe_time_anchors * len(unet_layers)
            elif self.use_positional_encoding == 0:
                self.encoder = BasicEncoder().cuda()
                self.input_dim = 2
            elif self.use_positional_encoding == 2:
                raise NotImplementedError()
            else:
                raise ValueError()

            self.input_layer = self.set_input_layer(len(unet_layers),
                                                    num_pe_time_anchors)
        else:
            self._set_positional_encoding()

        # define architecture
        if self.embedding_type == "object":
            self.set_net_object(num_unet_layers=len(unet_layers),
                                num_time_anchors=num_pe_time_anchors,
                                output_dim=output_dim)
        elif self.embedding_type == "view":
            self.arch_view_net = arch_view_net
            self.arch_view_mix_streams = arch_view_mix_streams

            if self.arch_view_disable_tl:
                self.input_dim = 0  # set_net_view functions will increase it

            self.set_net_view(num_unet_layers=len(unet_layers),
                              num_time_anchors=num_pe_time_anchors,
                              output_dim=output_dim)

        # options for 'unconstrained bypass' experiments.
        self.bypass_unconstrained = bypass_unconstrained
        if self.bypass_unconstrained:
            assert output_bypass
        self.name = placeholder_object_token if embedding_type == "object" else 'view'  # for debugging
        if 0:
            v = next(self.parameters())
            v.register_hook(lambda x: print(f"Computed backward in mapper [{self.name}]"))

    def set_net_object(self,
                       num_unet_layers: int,
                       num_time_anchors: int,
                       output_dim: int = 768):
        if self.original_ti:
            self.ti_embeddings = torch.nn.parameter.Parameter(
                self.original_ti_init_embed.unsqueeze(0), requires_grad=True)
            self.output_layer = nn.Identity()

        else:
            h_dim = self.arch_mlp_hidden_dims
            self.net = nn.Sequential(nn.Linear(self.input_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(h_dim, output_dim))

    def set_input_layer(self, num_unet_layers: int,
                        num_time_anchors: int) -> nn.Module:
        if self.use_positional_encoding:
            input_layer = nn.Linear(self.encoder.num_w * 2, self.input_dim)
            input_layer.weight.data = self.encoder.init_layer(
                num_time_anchors, num_unet_layers)
        else:
            input_layer = nn.Identity()
        return input_layer

    def forward(self,
                timestep: torch.Tensor,
                unet_layer: torch.Tensor,
                input_ids_placeholder_view: torch.Tensor,
                truncation_idx: int = None) -> MapperOutput:
        """
        Args:
        input_ids_placeholder_view: If embedding_type=='object', ignored. If 
            embedding_type=='view', use the token id to condition on the view 
            parameters for that token.
        """
        if self.original_ti or (self.embedding_type == "view"
                                and self.arch_view_net == 1):
            if self.embedding_type == "view":
                idx = [
                    self.lookup_ti_embedding[p.item()].item()
                    for p in input_ids_placeholder_view
                ]
                embedding = self.ti_embeddings[idx]
            else:
                embedding = self.ti_embeddings[[0] * len(timestep)]
            return embedding

        embedding = self.extract_hidden_representation(
            timestep, unet_layer, input_ids_placeholder_view)

        if self.use_nested_dropout:
            embedding = self.apply_nested_dropout(
                embedding, truncation_idx=truncation_idx)

        output = self.get_output(embedding)

        return output

    def get_encoded_input(self, timestep: torch.Tensor,
                          unet_layer: torch.Tensor) -> torch.Tensor:
        """ Encode the (t,l) params """
        encoded_input = self.encoder.encode(
            timestep,
            unet_layer,
        )  # (bs,2048)
        return self.input_layer(encoded_input)  # (bs, 160)

    def _prepare_view_token_param_lookup(self, rescale_min_max=False):
        """ 
        All possible view tokens that the mapper handles are known ahead of 
        time. This method prepares lookup dicts from tokenids (and tokens) to 
        the view parameters.

        The `rescale_min_max` thing: at training, we put all the training view 
        parameters in the range [-1,1]. If `rescale_min_max=True`, then we define
        the max and min parameters for that normalization based on the current 
        self.placeholder_view_tokens. So this should only be called when those 
        placeholder tokens are the same as what was done in training. 
        For inference, this is handled by the CheckpointHandler.load_mapper() 
        method (which calls the public-facing function `add_view_tokens_to_vocab`).
        But if this func is run with `rescale_min_max=True` with the wrong 
        placeholders set to self, then the generations will become weird
        """
        assert len(self.placeholder_view_tokens) == len(
            self.placeholder_view_token_ids)

        if 'dtu12d' not in self.placeholder_view_tokens[0]:
            assert all([
                s[:6] == '<view_' for s in self.placeholder_view_tokens
            ]), "not view tokens"

            view_params = [[
                string_to_num(num) for num in token[6:-1].split("_")
            ] for token in self.placeholder_view_tokens]

            self.view_token_2_view_params = dict(
                zip(self.placeholder_view_tokens, view_params))
            self.view_tokenid_2_view_params = dict(
                zip(self.placeholder_view_token_ids, view_params))

            view_params_all = torch.Tensor(view_params)
            if rescale_min_max:
                self.theta_min, self.theta_max = view_params_all[:, 0].min(
                ).item(), view_params_all[:, 0].max().item()
                self.phi_min, self.phi_max = view_params_all[:, 1].min().item(
                ), view_params_all[:, 1].max().item()
                self.r_min, self.r_max = view_params_all[:, 2].min().item(
                ), view_params_all[:, 2].max().item()

                if self.theta_min - self.theta_max == 0:
                    self.deg_freedom = "phi"
                else:
                    self.deg_freedom = "theta-phi"

        else:
            self.deg_freedom = "dtu-12d"

            # create lookups from {view_token,view_token_id} to cam_key and
            # camera parameters
            self.view_token_2_view_params = {}
            self.view_tokenid_2_view_params = {}
            self.view_token_2_cam_key = {}
            self.view_tokenid_2_cam_key = {}

            for (view_token,
                 view_token_id) in zip(self.placeholder_view_tokens,
                                       self.placeholder_view_token_ids):
                params, cam_key = TextualInversionDataset.dtu_token_to_cam_params(
                    view_token, cam_idx_as_int=True)
                self.view_token_2_view_params[view_token] = params
                self.view_tokenid_2_view_params[view_token_id] = params
                self.view_token_2_cam_key[view_token] = cam_key
                self.view_tokenid_2_cam_key[view_token_id] = cam_key

            # compute the normalizers if flagged. Only used when creating
            if rescale_min_max:
                # nb: the normalizing min/max params are computed wrt  *all the 
                # possible cameras* in the dtu dataset, and not the ones used in 
                # this particular model. This is so the mapper view ranges are 
                # consistent when re-using pretrained view tokens
                _, lookup_camidx_to_cam_params = TextualInversionDataset.dtu_generate_dset_cam_tokens_params()
                all_cams = torch.stack(list(lookup_camidx_to_cam_params.values()))
                self.cam_mins = all_cams.min(0).values.flatten()  # (12,)
                self.cam_maxs = all_cams.max(0).values.flatten()  # (12,)

    @staticmethod
    def scale_m1_1(x, xmin, xmax):
        """ scale a tensor to (-1,1). If xmin==xmax, do nothing"""
        if type(xmin) is not torch.Tensor:
            if xmin == xmax:
                return x
        return (x - xmin) / (xmax - xmin) * 2 - 1

    def get_view_params_from_token(self, input_ids_placeholder_view, device,
                                   dtype):
        """ 
        Given a set of token ids, `input_ids_placeholder_view`, that correspond
        to tokens like <"view_10_40_1p2>", return a tensor of the camera params,
        e.g. params (10,40,1.2). The dimension is (bs,3), where each batch element
        is 

        """
        # lookup the view parameters
        view_params = [
            self.view_tokenid_2_view_params[i.item()]
            for i in input_ids_placeholder_view
        ]

        if self.deg_freedom in ('phi', 'theta-phi'):
            # get individual params
            thetas, phis, rs = zip(*view_params)  # unzip the camera views
            thetas = torch.tensor(thetas, device=device,
                                  dtype=dtype).unsqueeze(1)
            phis = torch.tensor(phis, device=device, dtype=dtype).unsqueeze(1)
            rs = torch.tensor(rs, device=device, dtype=dtype).unsqueeze(1)

            # do scaling
            thetas = NeTIMapper.scale_m1_1(thetas, self.theta_min,
                                           self.theta_max)
            phis = NeTIMapper.scale_m1_1(phis, self.phi_min, self.phi_max)
            rs = NeTIMapper.scale_m1_1(rs, self.r_min, self.r_max)

            view_params = {'thetas': thetas, 'phis': phis, 'rs': rs}

        elif self.deg_freedom == "dtu-12d":
            cam_matrix = torch.stack([
                self.view_tokenid_2_view_params[tid.item()]
                for tid in input_ids_placeholder_view
            ]).to(device).to(dtype)
            cam_matrix = NeTIMapper.scale_m1_1(cam_matrix,
                                               self.cam_mins.to(device),
                                               self.cam_maxs.to(device))
            view_params = {'cam_matrix': cam_matrix}
        else:
            raise NotImplementedError()

        return view_params

    def get_encoded_view_input(self, input_ids_placeholder_view, device,
                               dtype):
        """ conditioning vectors """

        # get the view params from the view token id: (theta,phi,r)
        view_params = self.get_view_params_from_token(
            input_ids_placeholder_view, device=device, dtype=dtype)

        if self.deg_freedom == "phi":
            encoded_views = self.encode_phi(view_params['phis'])
        else:
            raise NotImplementedError(
                "this function was abandoned before I moved beyond phi-only variation"
            )

        return encoded_views

    def mix_encoding_and_views(self, encoded_tl: torch.Tensor,
                               encoded_views: torch.Tensor):
        if self.arch_view_mix_streams == 0:
            encoded_input_mixed = torch.cat((encoded_tl, encoded_views), dim=1)
        elif self.arch_view_mix_streams == 1:
            assert encoded_tl.shape == encoded_views.shape
            encoded_input_mixed = encoded_tl + encoded_views
        else:
            raise NotImplementedError

        return encoded_input_mixed

    def extract_hidden_representation(
            self, timestep: torch.Tensor, unet_layer: torch.Tensor,
            input_ids_placeholder_view: torch.Tensor) -> torch.Tensor:

        # for backcompatibility, this is how the old experiments were handled
        if self.arch_view_net <= 14:
            if self.embedding_type == "object":
                encoded_input_tl = self.get_encoded_input(timestep, unet_layer)
                embedding = self.net(encoded_input_tl)

            elif self.embedding_type == "view":
                device, dtype = input_ids_placeholder_view.device, timestep.dtype
                encoded_input_views = self.get_encoded_view_input(
                    input_ids_placeholder_view, device, dtype)

                if self.arch_view_disable_tl:
                    embedding = self.net(encoded_input_views)
                else:
                    encoded_input_tl = self.get_encoded_input(
                        timestep, unet_layer)
                    encoded_input_mixed = self.mix_encoding_and_views(
                        encoded_input_tl, encoded_input_views)
                    embedding = self.net(encoded_input_mixed)
            else:
                raise ValueError()
        # other experiments were using the Fourier features positional encoding
        else:
            encoded_input = self.do_positional_encoding(
                timestep, unet_layer, input_ids_placeholder_view)
            embedding = self.net(encoded_input)

        return embedding

    def apply_nested_dropout(self,
                             embedding: torch.Tensor,
                             truncation_idx: int = None) -> torch.Tensor:
        if self.training:
            if random.random() < self.nested_dropout_prob:
                dropout_idxs = torch.randint(low=0,
                                             high=embedding.shape[1],
                                             size=(embedding.shape[0], ))
                for idx in torch.arange(embedding.shape[0]):
                    embedding[idx][dropout_idxs[idx]:] = 0
        if not self.training and truncation_idx is not None:
            for idx in torch.arange(embedding.shape[0]):
                embedding[idx][truncation_idx:] = 0
        return embedding

    def get_output(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding = self.output_layer(embedding)

        # split word embedding and output bypass (if enabled) and save to object
        if not self.output_bypass:
            output = MapperOutput(word_embedding=embedding,
                                  bypass_output=None,
                                  bypass_unconstrained=False,
                                  output_bypass_alpha=self.output_bypass_alpha)
        else:
            dim = embedding.shape[1] // 2
            output = MapperOutput(
                word_embedding=embedding[:, :dim],
                bypass_output=embedding[:, dim:],
                bypass_unconstrained=self.bypass_unconstrained,
                output_bypass_alpha=self.output_bypass_alpha)

        # apply norm scaling to the word embedding (if enabled)
        if self.norm_scale is not None:
            output.word_embedding = F.normalize(output.word_embedding,
                                                dim=-1) * self.norm_scale

        return output

    def add_view_tokens_to_vocab(self, placeholder_view_tokens_new: List[str],
                                 placeholder_view_token_ids_new: List[int]):
        """ 
        Add new tokens to the vocabulary. 
        This is intended to be called by external scripts doing inference with 
        novel view tokens (tokens that were not used in training).
        """
        assert len(placeholder_view_tokens_new) == len(
            placeholder_view_token_ids_new)

        ## get ids of tokens that are novel
        mask_new_tokens = ~np.isin(np.array(placeholder_view_tokens_new),
                                   np.array(self.placeholder_view_tokens))
        mask_new_token_ids = ~np.isin(
            np.array(placeholder_view_token_ids_new),
            np.array(self.placeholder_view_token_ids))
        assert np.all(np.all(mask_new_tokens == mask_new_token_ids))
        idxs_new_tokens = np.where(mask_new_tokens)[0]

        # add the new tokens to the index
        self.placeholder_view_tokens += [
            placeholder_view_tokens_new[i] for i in idxs_new_tokens
        ]
        self.placeholder_view_token_ids += [
            placeholder_view_token_ids_new[i] for i in idxs_new_tokens
        ]

        # recreate the lookup table WITHOUT rescaling the ranges of the MLP
        self._prepare_view_token_param_lookup(rescale_min_max=False)

    def _set_positional_encoding(self):
        """ Set up the Fourier features positional encoding for t,l, and pose 
        params. 
        There are two ways to combine encodings: (i) adds the frequencies of the 
        different params (like in the Fourier Fetaures paper). (ii) computes a 
        fourier feature for each term independently, and then concats them, with 
        an optional normalization. 

        Warning: the `seed` argument to `FourierPositionalEncodingNDims` is important 
        to reloading old 
        """
        if self.arch_view_disable_tl:
            raise NotImplementedError(
                "For arch_view_net > 14, assume tl conditioning always")

        # set up the variance of the random fourier feature frequencies
        sigmas = [self.pe_sigmas.sigma_t, self.pe_sigmas.sigma_l]
        if self.embedding_type == "object":
            pass

        elif self.embedding_type == "view":

            # warning: order of sigmas must match do_positional_encoding implementation
            if self.deg_freedom == "phi":
                sigmas += [self.pe_sigmas.sigma_phi]

            elif self.deg_freedom == "theta-phi":
                sigmas += [
                    self.pe_sigmas.sigma_theta, self.pe_sigmas.sigma_phi
                ]
            elif self.deg_freedom == "dtu-12d":
                sigmas += [self.pe_sigmas.sigma_dtu12] * 12
            else:
                raise NotImplementedError()

        # lookup the positional encoding dimension
        self.pose_encode_dim = {
            '15': {
                'object': 64,
                'view': 64
                }
        }[str(self.arch_view_net)][self.embedding_type]

        # generate the positional encoder
        if self.arch_view_net in (15, 18, 20, 22):
            self.positional_encoding_method = "add_freqs"
            self.input_dim = self.pose_encode_dim
            self.encoder = FourierPositionalEncodingNDims(
                dim=self.pose_encode_dim, sigmas=sigmas, seed=0)

        elif self.arch_view_net in (16, 17, 19, 21):
            self.positional_encoding_method = "concat_features"
            self.input_dim = self.pose_encode_dim * len(sigmas)
            self.normalize = {
                '16': False,
                '17': True,
                '19': False,
                '21': False
            }[str(self.arch_view_net)]
            self.encoder = [
                FourierPositionalEncodingNDims(dim=self.pose_encode_dim,
                                               sigmas=[sigma],
                                               seed=i,
                                               normalize=self.normalize)
                for i, sigma in enumerate(sigmas)
            ]

        else:
            raise NotImplementedError(
                "Need to define the pos encoding combination method for ",
                f"for arch_view_net=[self.arch_view_net] (in this function)")

    def do_positional_encoding(self, timestep, unet_layer,
                               input_ids_placeholder_view):
        """ new methods for getting positional encoding for self.arch_view>=14"""
        # put timestep and unet_layer in range [-1,1]
        timestep = timestep / 1000 * 2 - 1
        unet_layer = unet_layer / self.num_unet_layers * 2 - 1
        data = torch.stack((timestep, unet_layer), dim=1)

        # if it's a view-mapper, then add view data
        if self.embedding_type == "view":
            device, dtype = timestep.device, timestep.dtype
            view_params = self.get_view_params_from_token(
                input_ids_placeholder_view, device, dtype)

            if self.deg_freedom == "phi":
                data = torch.cat((data, view_params['phis']), dim=1)
            elif self.deg_freedom == "theta-phi":
                data = torch.cat(
                    (data, view_params['thetas'], view_params['phis']), dim=1)
            elif self.deg_freedom == "dtu-12d":
                data = torch.cat((data, view_params['cam_matrix']), dim=1)
            else:
                raise NotImplementedError()

        # do pos encoding (explanation in docstring for _set_positional_encoding)
        if self.positional_encoding_method == "add_freqs":
            encoding = self.encoder(data)

        elif self.positional_encoding_method == "concat_features":
            encoding = torch.cat(
                [self.encoder[i](data[:, i]) for i in range(data.shape[1])],
                dim=1)

        else:
            raise ValueError()

        return encoding

    def set_net_view(self,
                     num_unet_layers: int,
                     num_time_anchors: int,
                     output_dim: int = 768):
        # Original-TI (also has arch-code-1)
        if self.original_ti or self.arch_view_net == 1:
            # baseline - TI baseline, which is one thing no matter what.
            assert self.original_ti_init_embed is not None
            if self.output_bypass:
                raise
            self.ti_embeddings = self.original_ti_init_embed.unsqueeze(
                0).repeat(len(self.placeholder_view_token_ids), 1)
            self.ti_embeddings = torch.nn.parameter.Parameter(
                self.ti_embeddings.clone(), requires_grad=True)
            # self.ti_embeddings.register_hook(lambda x: print(x))
            self.lookup_ti_embedding = dict(
                zip(self.placeholder_view_token_ids,
                    torch.arange(len(self.placeholder_view_token_ids))))
            self.output_layer = nn.Identity()  # the MLP aready does projection

        
        # this architecture key 15 is the final model used in the paper
        elif self.arch_view_net in (15,):
            h_dim = 64
            self.net = nn.Sequential(nn.Linear(self.input_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(h_dim, output_dim))
        
        else:
            raise NotImplementedError()
