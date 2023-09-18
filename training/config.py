from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Union
from training.pretrained_models import lookup_pretrained_models

import ipdb
from constants import VALIDATION_PROMPTS
from utils.types import PESigmas


@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Name of experiment. This will be the name of the output folder. If left "", uses config file name
    exp_name: str = ""
    # check if overwriting
    overwrite_ok: bool = False
    # The output directory where the model predictions and checkpoints will be written
    exp_dir: Path = Path("./outputs")
    # Save interval
    save_steps: int = 1000
    # [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
    # `output_dir/runs/**CURRENT_DATETIME_HOSTNAME`
    logging_dir: Path = Path("logs")
    # The integration to report the results to. Supported platforms are "tensorboard" '
    # (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    report_to: str = "all"
    # Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator`
    checkpoints_total_limit: Optional[int] = None
    # Log dataset
    save_dataset_images: bool = True


@dataclass
class DataConfig:
    """ Parameters for data """
    # A folder containing the training data
    train_data_dir: Path
    # for learnable mode 3, a list of subdirs of train_data_dir that have the separate object datasets
    train_data_subsets: List[Path] = None
    # A token to use as a placeholder for the concept
    placeholder_object_token: str = "<>"
    # Super category token to use for normalizing the mapper output
    super_category_object_token: Optional[str] = "object"
    super_category_view_token: Optional[str] = "view"
    # some list versions of these things for learnable mode 3
    placeholder_object_tokens: List[str] = None
    super_category_object_tokens: Optional[List[str]] = None
    # if learnable_mode==1 (learning views only), object token. Either a string or path to a NeTI mapper.
    fixed_object_token_or_path: Union[str, Path] = None
    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process
    dataloader_num_workers: int = 8
    # How many times to repeat the training data
    repeats: int = 100
    # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
    resolution: int = 512
    # preprocessing dtu. 0=padding+resizing, 1=resizing to (512,384),
    dtu_preprocess_key: int = 1
    # Whether to center crop images before resizing to resolution
    center_crop: bool = False
    # data augmentation flip rate. If learning a non-object tokens, automatically set to 0
    flip_p: float = 0.5
    # this is a placeholder that will be filled at runtime
    placeholder_view_tokens = None
    # see training/dataset.py
    caption_strategy: int = 0
    # how the camera params are represented {"spherical", "12"}
    camera_representation: str = "spherical"
    dtu_lighting: str = 3
    # which dtu images to include in train views. 0=all, -1=idxs(12,36,1),
    # -2=range(12,26,2)
    # {3,6,9} are the views hardcoded into standard RegNerf setup
    dtu_subset: int = -2
    # data augmentations
    augmentation_key: int = 0


@dataclass
class ModelConfig:
    """ Parameters for defining all models """
    # Path to pretrained model or model identifier from huggingface.co/models
    pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
    # set pretrained view-mapper ... either path or a lookup key. 
    # Exactly one of these must be not None for learnable modes (3,4), ow `post_init` throwds error
    pretrained_view_mapper: Path = None
    pretrained_view_mapper_key: int = None
    # dimension of word embedding. Is 768 if sd1 and 1024 if sd2
    word_embedding_dim: int = 768
    # dimension of hidden layers in the MLP
    arch_mlp_hidden_dims: int = 128
    # Whether to use our Nested Dropout technique
    use_nested_dropout: bool = True
    # Probability to apply nested dropout during training
    nested_dropout_prob: float = 0.5
    # Whether to normalize the norm of the mapper's output vector
    normalize_object_mapper_output: bool = True
    normalize_view_mapper_output: bool = False
    # Target norm for the mapper's output vector
    target_norm_object: float = None
    target_norm_view: float = None
    # Pos encoding for (t,l) conditioning. 0 - scale to [-1,1], 1 - pos encoding
    # proposed by Neti. 2 - standard Fourier feature pos encoding.
    use_positional_encoding_object: int = 1
    use_positional_encoding_view: int = 1
    # Sigmas used for computing positional encoding
    pe_sigmas: Dict[str, float] = field(
        default_factory=lambda: {
            'sigma_t': 0.03,
            'sigma_l': 2.0,
            'sigma_theta': 1.0,
            'sigma_phi': 1.0,
            'sigma_r': 1.0,
            'sigma_dtu12': 2.0,
        })
    pe_sigma_exp_key: int = 0
    pe_t_exp_key: int = 0 
    pe_l_exp_key: int = 0 
    pe_sigmas_view: Dict[str, float] = field(
        default_factory=lambda: {'sigma_phi': 1.0})
    # Number of time anchors for computing our positional encodings
    num_pe_time_anchors: int = 10
    # Whether to output the textual bypass vector
    output_bypass_object: bool = True
    output_bypass_view: bool = True
    # Revision of pretrained model identifier from huggingface.co/models
    revision: Optional[str] = None
    # Whether training should be resumed from a previous checkpoint.
    mapper_checkpoint_path: Optional[Path] = None
    # configuration for view neti-mappers. Ignored by object neti-mappers
    arch_view_net: int = 0
    arch_view_mix_streams: int = 0
    arch_view_disable_tl: bool = True
    # Run as original-TI. A single vector is learned per placeholder token.
    original_ti: bool = False
    # free movement in the bypass space
    bypass_unconstrained_object: bool = False
    bypass_unconstrained_view: bool = False
    # size of alpha hyperparameter for the output bypass
    output_bypass_alpha_view: float = 0.2
    output_bypass_alpha_object: float = 0.2

    def __post_init__(self):
        if self.pe_sigmas is not None:
            self.pe_sigmas = PESigmas(
                sigma_t=self.pe_sigmas['sigma_t'],
                sigma_l=self.pe_sigmas['sigma_l'],
                sigma_theta=self.pe_sigmas.get('sigma_phi', 1.0),
                sigma_phi=self.pe_sigmas.get('sigma_phi', 1.0),
                sigma_r=self.pe_sigmas.get('sigma_phi', 1.0),
                sigma_dtu12=self.pe_sigmas.get('sigma_dtu12', 2.0))
            if self.pe_sigma_exp_key == 1:
                self.pe_sigmas.sigma_dtu12 = 1.0
            elif self.pe_sigma_exp_key == 2:
                self.pe_sigmas.sigma_dtu12 = 0.5
            elif self.pe_sigma_exp_key == 3:
                self.pe_sigmas.sigma_dtu12 = 0.25
            elif self.pe_sigma_exp_key == 4:
                self.pe_sigmas.sigma_dtu12 = 0.75
            elif self.pe_sigma_exp_key == 5:
                self.pe_sigmas.sigma_dtu12 = 0.1

            if self.pe_t_exp_key == 0:
                self.pe_sigmas.sigma_t = 0.03
            elif self.pe_t_exp_key == 1:
                self.pe_sigmas.sigma_t = 0.06
            elif self.pe_t_exp_key == 2:
                self.pe_sigmas.sigma_t = 0.2
            elif self.pe_t_exp_key == 3:
                self.pe_sigmas.sigma_t = 0.5
            else: 
                raise
            
            if self.pe_l_exp_key == 0:
                self.pe_sigmas.sigma_l = 2.0
            elif self.pe_l_exp_key == 1:
                self.pe_sigmas.sigma_l = 4.0
            else: 
                raise 


@dataclass
class EvalConfig:
    """ Parameters for validation """
    # Prompts for validation (only for learnable_mode=0)
    validation_prompts: List[str] = field(
        default_factory=lambda: VALIDATION_PROMPTS)
    # Prompts for valiation (only for learnable_mode>0)
    validation_view_tokens = None
    # Number of images that should be generated during validation with `validation_prompt`
    num_validation_images: int = 3
    # Seeds to use for generating the validation images
    validation_seeds: Optional[List[int]] = field(
        default_factory=lambda: [0, 1, 2])
    # Run validation every X steps.
    validation_steps: int = 250
    # Number of denoising steps
    num_denoising_steps: int = 30
    # upsampling strategy
    dtu_upsample_key: int = 1
    # for learnable_mode==3, which of the `placholder_object_tokens` to include in validation
    eval_placeholder_object_tokens: List[str] = None

    def __post_init__(self):
        if self.validation_seeds is None:
            self.validation_seeds = list(range(self.num_validation_images))
        assert len(self.validation_seeds) == self.num_validation_images, \
            "Length of validation_seeds should equal num_validation_images"


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Total number of training steps to perform.
    max_train_steps: Optional[int] = 1_000
    # Learning rate
    learning_rate: float = 1e-3
    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size
    scale_lr: bool = True
    # Batch size (per device) for the training dataloader
    train_batch_size: int = 3
    # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass
    gradient_checkpointing: bool = False
    # Number of updates steps to accumulate before performing a backward/update pass
    gradient_accumulation_steps: int = 3
    # A seed for reproducible training
    seed: Optional[int] = None
    # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",
    # "constant", "constant_with_warmup"]
    lr_scheduler: str = "constant"
    # Number of steps for the warmup in the lr scheduler
    lr_warmup_steps: int = 0
    # The beta1 parameter for the Adam optimizer
    adam_beta1: float = 0.9
    # The beta2 parameter for the Adam optimizer
    adam_beta2: float = 0.999
    # Weight decay to use
    adam_weight_decay: float = 1e-2
    # Epsilon value for the Adam optimizer
    adam_epsilon: float = 1e-08
    # Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.
    # and an Nvidia Ampere GPU.
    mixed_precision: str = "no"
    # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32: bool = False


@dataclass
class RunConfig:
    """ The main configuration for the coach trainer """
    # learnable mode says what kind of token combination (object, view, mix) we're learning. See dataset.py docs.
    # Modes:
    #  0 object only
    #  1 view only
    #  2 view and object (optionally, object is pretrained)
    #  3 view and multiple objects
    #  4 view and object where view is pretrained. Prompts same as 2.
    #  5 view and object where view is pretrained, and frozen. Prompts same as 2.
    learnable_mode: int = 0
    debug: bool = False
    seed: int = 0
    log: LogConfig = field(default_factory=LogConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def __post_init__(self):
        if self.optim.train_batch_size > 3:
            raise ValueError(
                "batch size should be 3 and so should grad accumulation")
        # do some error checking
        if self.learnable_mode == 3:
            assert self.data.dataloader_num_workers == 0, "can't support multiple workers right now for learnable mode 3"
            assert self.data.super_category_object_tokens is not None
            if self.eval.eval_placeholder_object_tokens is not None:
                assert all(
                    [
                        d in self.data.placeholder_object_tokens
                        for d in self.eval.eval_placeholder_object_tokens
                    ]
                ), "eval.eval_placeholder_tokens not in data.placeholder_object_tokens"
        if self.data.placeholder_object_tokens is not None:
            assert len(self.data.placeholder_object_tokens) == len(
                set(self.data.placeholder_object_tokens)
            ), "cfg.data.placeholder_object_tokens must be unique strings"

        if self.learnable_mode in (4, 5):
            # either the path is set OR the key is set, but not both
            assert self.model.pretrained_view_mapper or self.model.pretrained_view_mapper_key
            # assert not (self.model.pretrained_view_mapper and self.model.pretrained_view_mapper_key)
            if self.model.pretrained_view_mapper_key:
                self.model.pretrained_view_mapper = lookup_pretrained_models[str(self.model.pretrained_view_mapper_key)]




