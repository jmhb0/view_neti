import ipdb
import random
from pathlib import Path
from typing import Dict, Any, Union, Literal, List

import PIL
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image, ImageOps
from packaging import version
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as T
from transformers import CLIPTokenizer

from constants import IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_SMALL, PATH_DTU_CALIBRATION_DIR, DTU_SPLIT_IDXS
from utils.utils import num_to_string, string_to_num, filter_paths_imgs

if version.parse(version.parse(
        PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


class TextualInversionDataset(Dataset):

    def __init__(self,
                 data_root: Path,
                 tokenizer: CLIPTokenizer,
                 camera_representation: Literal["spherical", "dtu-12d"],
                 learnable_mode: int,
                 train_data_subsets: List[Path] = None,
                 placeholder_object_tokens: List[str] = None,
                 fixed_object_token_or_path: Union[Path, str] = None,
                 size: int = 768,
                 repeats: int = 100,
                 interpolation: str = "bicubic",
                 flip_p: float = 0.0,
                 set: str = "train",
                 placeholder_object_token: str = "*",
                 dtu_lighting: str = "3",
                 dtu_subset: int = 0,
                 caption_strategy: int = 0,
                 dtu_preprocess_key: int = 0,
                 augmentation_key: int = 0,
                 center_crop: bool = False):
        """
        fixed_object

        learnable_mode: integer key for what mode of caption we need to generate. 
            In caption examples things in `<>` are learnable.
            0: object only, "A photo of a <object>".
            1: view only, "<view_x>. A photo of a {object}". Where {object} is 
            predefined as either a string or a pretrained NeTI mapper.
            2: view and object learned jointly. "<view_x>. A photo of a <object>".
            3: multi-scene training. Learn one view-mapper shared accross scenes 
                and an object mapper per-scene. "<view_x>. A photo of a <object_y>".
            4: use a pretrained view-mapper (probably pretrained from mode 3) and
                also train a new object mapper on one scene. Both view- and object-maper
                are learnable.
            5: same as mode 4, except the view mapper is not learnable. .

        fixed_object_token_or_path: if learnable_mode==1, then this cannot be None. Either a 
            string if using a word from the existing vocabulary, or if using a pretrained
            viewNeTI-mapper, then write the path to that mapper. 
        dtu_preprocess_key: what image preprocessing for dtu dataset. 
        """
        self.learnable_mode = learnable_mode
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_object_token = placeholder_object_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.train_data_subsets = train_data_subsets
        self.camera_representation = camera_representation
        self.dtu_lighting = dtu_lighting
        self.dtu_subset = dtu_subset
        self.dtu_preprocess_key = dtu_preprocess_key
        print("dtu process key ", dtu_preprocess_key)
        if self.learnable_mode != 0:
            self.flip_p = 0

        # load image paths. Mode 3 is a special case that has multipble object training datasets.
        if self.learnable_mode != 3:
            # get fnames, and filter for png
            self.image_paths = list(self.data_root.glob("*"))
            self.image_paths = filter_paths_imgs(self.image_paths)

            # dtu-specific edits for lighting and non-excluded cam idxs
            if self.camera_representation in ('dtu-12d') and self.learnable_mode != 0:
                self.image_paths = TextualInversionDataset.dtu_filter_fnames_lighting(
                    self.image_paths, self.dtu_lighting)
                idxs = TextualInversionDataset.dtu_get_train_idxs(dtu_subset)
                self.image_paths = TextualInversionDataset.dtu_filter_image_paths_from_idx(
                    self.image_paths, idxs)
            self.num_images = len(self.image_paths)
            self.image_paths_flattened = self.image_paths

        else:
            # get lookup of image paths
            self.image_paths = {}
            self.num_images = 0
            # iterate over the subdirs, get fnames, and do filtering
            for subdir in self.train_data_subsets:
                # get fnames, and filter for png
                subdir = str(subdir)  # ow the key will be PosixPath
                self.image_paths[subdir] = list(
                    (self.data_root / subdir).glob("*"))
                self.image_paths[subdir] = filter_paths_imgs(
                    self.image_paths[subdir])

                # dtu-specific edits for lighting and non-excluded cam idxs
                if self.camera_representation in ('dtu-12d'):
                    self.image_paths[
                        subdir] = TextualInversionDataset.dtu_filter_fnames_lighting(
                            self.image_paths[subdir], self.dtu_lighting)
                    idxs = TextualInversionDataset.dtu_get_train_idxs(
                        dtu_subset)
                    self.image_paths[
                        subdir] = TextualInversionDataset.dtu_filter_image_paths_from_idx(
                            self.image_paths[subdir], idxs)

                assert len(self.image_paths[subdir]) > 0

            # put all the image_paths in a single list, which is handy later
            self.image_paths_flattened = [
                p for row in list(self.image_paths.values()) for p in row
            ]
            self.num_images = len(self.image_paths_flattened)
            # initialize (see docstring for `reset_sampled_object` in this class)
            self.current_object_idx = np.random.choice(
                len(self.train_data_subsets))

        assert self.num_images > 0, "no .png images found. Check the --data.train_data_dir option" 
        self._length = self.num_images
        print(f"Running on {self.num_images} images")
        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = IMAGENET_TEMPLATES_SMALL
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.caption_strategy = caption_strategy
        if self.caption_strategy > 0:
            assert self.learnable_mode == 1, "alt caption_strategy only implemented for mode 1"

        if self.learnable_mode == 0:
            self.placeholder_object_tokens = [self.placeholder_object_token]
            self.placeholder_view_tokens = []
            self.fixed_object_token = None  # not applicable to this case

        elif self.learnable_mode in (1, 2, 3, 4, 5):
            if self.learnable_mode in (3, 4, 5):
                assert self.camera_representation == 'dtu-12d', "Haven't verified whether these modes work outside dtu dataset"

            self.placeholder_view_tokens = self.generate_view_tokens()
            self.placeholder_view_tokens = self.order_view_tokens(
                self.placeholder_view_tokens)

            # check if preloading a Neti-learned object; if so, then it's a 'placeholder_object_token'
            if Path(fixed_object_token_or_path).suffix == ".pt":
                self.fixed_object_token_pretrained = True
                cfg = torch.load(fixed_object_token_or_path)['cfg']
                placeholder_object_token_saved = cfg['data'][
                    'placeholder_object_token']
                if placeholder_object_token_saved != placeholder_object_token:
                    raise ValueError(
                        f"Saved Neti placeholder token is [{placeholder_object_token_saved}]",
                        f", different from config, [{placeholder_object_token}]"
                    )

                self.fixed_object_token = self.placeholder_object_token
                self.placeholder_object_tokens = [
                    self.placeholder_object_token
                ]

            else:
                if learnable_mode == 1:
                    self.fixed_object_token_pretrained = False
                    self.fixed_object_token = fixed_object_token_or_path
                    self.placeholder_object_tokens = []

                elif learnable_mode in (2, 4, 5):
                    self.fixed_object_token_pretrained = False
                    self.fixed_object_token = None
                    self.placeholder_object_tokens = [
                        self.placeholder_object_token
                    ]
                elif learnable_mode == 3:
                    self.fixed_object_token_pretrained = False
                    self.fixed_object_token = None
                    self.placeholder_object_tokens = placeholder_object_tokens
                    # map from object id (like `scan25`) to object token like `<statue>`. Used in getitem
                    self.lookup_object_to_placeholder_object_token = {
                        str(s): t
                        for (s, t) in zip(self.train_data_subsets,
                                          self.placeholder_object_tokens)
                    }

        else:
            raise ValueError()

        self.placeholder_tokens = self.placeholder_view_tokens + self.placeholder_object_tokens
        assert hasattr(self, 'placeholder_object_tokens')
        assert hasattr(self, 'placeholder_view_tokens')

        self.augmentation_key = augmentation_key

        if self.augmentation_key > 0:
            if self.learnable_mode == 0:
                size = (self.size,self.size)
            elif self.dtu_preprocess_key == 0:
                size = (512, 512)
            elif self.dtu_preprocess_key == 1:
                size = (384, 512)  # size axes are reversed compared to PIL oib

            if self.augmentation_key == 1:
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)],
                                  p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                    T.RandomApply([T.RandomRotation(degrees=10, fill=1)],
                                  p=0.75),
                    T.RandomResizedCrop(size, scale=(0.850, 1.15)),
                ])
            elif self.augmentation_key == 2:
                # the same but without the geometric transforms
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)],
                                  p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                ])
            elif self.augmentation_key == 3:
                # the same but minus the small geometric transforms
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)],
                                  p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                    T.RandomApply([T.RandomRotation(degrees=10, fill=1)],
                                  p=0.75),
                ])
            elif self.augmentation_key == 4:
                # the same but minus the small geometric transforms
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)],
                                  p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                    T.RandomResizedCrop(size, scale=(0.850, 1.15)),
                ])
            elif self.augmentation_key == 5:
                # the same but the small geometric transforms are now very small
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)],
                                  p=0.75),
                    # T.RandomGrayscale(p=0.1),
                    T.RandomApply(
                        [T.GaussianBlur(5, (0.1, 0.2))],
                        p=0.25),  # a little higher (no grayscale anymore)
                    T.RandomResizedCrop(size,
                                        scale=(0.950, 1.05)),  # very small now
                ])
            elif self.augmentation_key == 6:
                # the exact thing used in RealFusion
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)],
                                  p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10),
                    T.RandomApply([T.RandomRotation(degrees=10, fill=1)],
                                  p=0.75),
                    T.RandomResizedCrop(size, scale=(0.70, 1.3)),
                ])
            elif self.augmentation_key == 7:
                # same as RealFusion but no grayscale ... I turn up the gaussianblur probablity a bit
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)],
                                  p=0.75),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.2),
                    T.RandomApply([T.RandomRotation(degrees=10, fill=1)],
                                  p=0.75),
                    T.RandomResizedCrop(size, scale=(0.70, 1.3)),
                ])

            elif self.augmentation_key == 8:
                # the same as 7 but without the geometric transforms. For ablations
                self.augmentations = T.Compose([
                    T.RandomApply([T.ColorJitter(0.04, 0.04, 0.04, 0.04)],
                                  p=0.75),
                    T.RandomGrayscale(p=0.1),
                    T.RandomApply([T.GaussianBlur(5, (0.1, 0.2))], p=0.10)
                ])

            else:
                raise

    @staticmethod
    def dtu_get_train_idxs(dtu_subset):
        """ 
        For DTU dataset, get the cam-idxs defined based on the `dtu_subset` key,
        which is in the config.

        '0' is the 'full' set up to 49 (though some scans have more)
        {3,6,9} are the splits defined in RegNerf and used by other sparse nerf 
        methods. 
        {-1,-2,-3} are splits I made up for idxs between (12,36). -1 is all of thosee
        idxs, -2 is the same but skipping every 2nd, and -3 skips every 3rd image.
        This idx range has a medium pitch angle (not from 
        above and not from the side, but in between). But it also includes imgs 
        that are in the 'exclude' list from RegNerf code bc of bad imaging 
        quality, so might not want to use them.
        """
        if dtu_subset == 0:
            idxs = DTU_SPLIT_IDXS['train'] + DTU_SPLIT_IDXS['test']
        elif dtu_subset == 1:
            idxs = DTU_SPLIT_IDXS['train'][:1]
        elif dtu_subset == 3:
            idxs = DTU_SPLIT_IDXS['train'][:3]
        elif dtu_subset == 6:
            idxs = DTU_SPLIT_IDXS['train'][:6]
        elif dtu_subset == 9:
            idxs = DTU_SPLIT_IDXS['train']
        elif dtu_subset == -1:
            idxs = list(range(12, 36))
        elif dtu_subset == -2:
            idxs = list(range(12, 36, 2))
        elif dtu_subset == -3:
            idxs = list(range(12, 36, 3))
        else:
            raise NotImplementedError()

        return idxs

    @staticmethod
    def dtu_filter_fnames_lighting(image_paths, dtu_lighting):
        """ Choose only image pathe with one lighting type. """
        return [f for f in image_paths if f.stem.split("_")[2] == dtu_lighting]

    @staticmethod
    def dtu_cam_info_from_fname(fname):
        """ 
        Extract the cam_idx and lighting_idx from the filename. 
        The cameras are 0-indexed but their fnames are 1-indexed (bc prior work
        did that). I'm putting this inside a wrapper to make sure this '-1' change
        happens without having to remember each time I need it.
        """
        fname = Path(fname)
        cam_idx, lighting_idx = fname.stem.split("_")[1:3]
        cam_idx = int(cam_idx) - 1
        return cam_idx, lighting_idx

    @staticmethod
    def dtu_cam_and_lighting_to_fname(cam_idx, lighting_idx):
        """ 
        Again, making this a wrapper to make sure the '-1' happens without 
        having to remember each time.
        """
        cam_idx += 1
        return f"rect_{cam_idx:03d}_{lighting_idx}_r5000.png"

    @staticmethod
    def dtu_filter_image_paths_from_idx(image_paths, idxs):
        """ 
        Filter dtu image paths by list of camera / view idxs.
        WARNING: the indexing is 0-indexed, while the filenames are 1-indexd (a 
        convention inherited from past projects).
        Return image paths sorted according to view index.
        """

        def get_cam_number(file_path):
            """ extract cam idx from file (subtract 1 to map 1-idx to 0-idx)"""
            return TextualInversionDataset.dtu_cam_info_from_fname(
                file_path)[0]

        # two cases: a list, or a dict of lists
        # if type(image_paths) != dict:
        image_paths = [f for f in image_paths if get_cam_number(f) in idxs]
        image_paths = sorted(image_paths, key=get_cam_number)

        # else:
        #     for k in image_paths.keys():
        #         image_paths[k] = [f for f in image_paths[k] if get_cam_number(f) in idxs]
        #         image_paths[k] = sorted(image_paths[k], key=get_cam_number)

        return image_paths

    def generate_view_tokens(self):
        """ Generate view tokens from the filenames.
        For spherical coordinates, its <view_{t}_{p}_{r}> where (t,p,r) are:
            t: theta, polar angle. 
            p: phi, azimuth angle.
            r: radius.
        For dtu12, its a similar idea (see the code)
        """
        if self.camera_representation == "spherical":
            view_prefixes = [f.stem.split("___")[-1] for f in self.image_paths]
            assert all([len(vp.split("_")) == 3 for vp in view_prefixes])
            placeholder_view_tokens = [f"<view_{v}>" for v in view_prefixes]
            placeholder_view_tokens = list(
                set(placeholder_view_tokens))  # dedupe

        elif self.camera_representation == "dtu-12d":
            # Get tokens for all DTU cam viewps (64), even if the train set only
            # uses some of them. Create lookups to be used by __getitem__

            (self.lookup_camidx_to_view_token, self.lookup_camidx_to_cam_params
             ) = TextualInversionDataset.dtu_generate_dset_cam_tokens_params()
            self.lookup_view_token_to_camidx = {
                v: k
                for (k, v) in self.lookup_camidx_to_view_token.items()
            }

            # get all possible placeholder tokens, but then remove the tokens
            # for those other than
            image_paths = self.image_paths if self.learnable_mode != 3 else self.image_paths_flattened
            cam_idxs_dataset = [
                TextualInversionDataset.dtu_cam_info_from_fname(f)[0]
                for f in image_paths
            ]
            cam_idxs_dataset = np.unique(cam_idxs_dataset)
            placeholder_view_tokens = [
                self.lookup_camidx_to_view_token[k] for k in cam_idxs_dataset
            ]

        else:
            raise NotImplementedError(
                f"For cam representation [{self.camera_representation}]")

        return placeholder_view_tokens

    @staticmethod
    def dtu_cam_params_to_token(cam_params: torch.Tensor, cam_key='NULL'):
        """
        View token representation of a camera. `cam_key` is intended to match
        the idxs in the DTU dataset. If the view is novel (not in the DTU set), 
        then it can be whatever, which is why it's optional.
        Inverse of `dtu_token_to_cam_params`.
        """
        cam_params = cam_params.flatten()
        assert len(cam_params) == 12
        view_token = f"<view_dtu12d_cam{cam_key}_" + "_".join(
            [num_to_string(n.item(), tol=4) for n in cam_params]) + ">"

        return view_token

    @staticmethod
    def dtu_token_to_cam_params(view_token: str, cam_idx_as_int: bool = False):
        """
        Extract dtu-12d camera parameters and cam idx from a view token.
        `Inverse of dtu_cam_params_to_token`

        cam_idx_as_int bool: if True, cast cam_idx to int (not all tokens have 
        a castable-camidx - e.g. it might be 'null', so setting this True may 
        throw ValueError.
        """
        cam_idx = view_token.split("_")[2][3:]
        if cam_idx_as_int:
            cam_idx = int(cam_idx)

        cam_params = torch.tensor(
            [string_to_num(n) for n in view_token[:-1].split("_")[3:]])

        return cam_params, cam_idx


    @staticmethod
    def dtu_generate_dset_cam_tokens_params():
        """ 
        DTU dataset has 64(?) calibrated cameras. 
        Create lookup dictionaries mapping from these camera idxs to camera 
        params, view_tokens, and filenames.
        """
        fnames_cam = list(Path(PATH_DTU_CALIBRATION_DIR).iterdir())
        fnames_cam = [f for f in fnames_cam if f.suffix == ".txt"]

        # create a lookup img_idx->camera_params and idx->view_token
        lookup_camidx_to_cam_params = {}
        lookup_camidx_to_view_token = {}
        for f in fnames_cam:
            cam_key = int(f.stem.split("_")[1])
            cam_key -= 1  # send 1-indexed fname to 0-index key
            cam_params = TextualInversionDataset.read_text_file_to_tensor(f)
            assert cam_key not in lookup_camidx_to_cam_params.keys(
            ), f"key {k}"
            lookup_camidx_to_cam_params[cam_key] = cam_params
            view_token = TextualInversionDataset.dtu_cam_params_to_token(
                cam_params, cam_key)
            lookup_camidx_to_view_token[cam_key] = view_token

        return lookup_camidx_to_view_token, lookup_camidx_to_cam_params

    @staticmethod
    def read_text_file_to_tensor(file_path):
        """ for reading dtu camera calibration matrices """
        with open(file_path, 'r') as file:
            data = [[float(num) for num in line.strip().split()]
                    for line in file.readlines()]
        return torch.tensor(data)

    def order_view_tokens(self, placeholder_view_tokens):
        """ 
        `placeholder_view_tokens` is a list of view tokens. We want to assign
        an ordering which is helpful for generating validation images with that 
        same ordering. For example if we only vary the `phi` angle, then the 
        ordering is increasing phi.
         """
        if self.camera_representation == "spherical":
            view_params = torch.tensor(
                [[string_to_num(num) for num in token[6:-1].split("_")]
                 for token in placeholder_view_tokens])
            thetas, phis, rs = view_params[:,
                                           0], view_params[:,
                                                           1], view_params[:,
                                                                           2]
            n_thetas, n_phis, n_rs = len(torch.unique(thetas)), len(
                torch.unique(phis)), len(torch.unique(rs))

            if n_thetas == 1 and n_phis > 1 and n_rs == 1:
                # phi-only case
                idxs = torch.argsort(phis)
            elif n_thetas > 1 and n_phis > 1 and n_rs == 1:
                thetas_uniq = torch.unique(thetas).sort().values
                phis_uniq = torch.unique(phis).sort().values

                def evenly_spaced_sample(lst, k):
                    if type(lst) is torch.Tensor:
                        lst = list([l.item() for l in lst])
                    n = len(lst)
                    if k >= n:
                        return lst
                    indices = [int(i * (n - 1) / (k - 1)) for i in range(k)]
                    return [lst[i] for i in indices]

                idxs = torch.arange(len(view_params))

            else:
                err = f"New 'degree of freedom' setting. Need to define a new ordering"
                err += f"View params are {view_params}"
                raise NotImplementedError(err)

            # case where we only have `phi` variation

            placeholder_view_tokens = [
                placeholder_view_tokens[i] for i in idxs
            ]

        elif self.camera_representation == "dtu-12d":
            # order based on the camera index, which is the same as the key
            keys = [
                self.lookup_view_token_to_camidx[t]
                for t in placeholder_view_tokens
            ]
            keys.sort()
            placeholder_view_tokens = [
                self.lookup_camidx_to_view_token[k] for k in keys
            ]

        return placeholder_view_tokens

    def reset_sampled_object(self):
        """ 
        *** No longer using this function ****

        For learnable_mode==3, we train multipe object tokens. If we have some 
        batch_size, but gradient_accumulation_steps>1, we might want alll the 
        samples in the same set of accumulated batches to be from the same scene
         so that the object-token gradients are not too noisy. 

        To do that, we sample the object indexed by `self.current_object_idx`, 
        and then we change this value only after accumuation. 
        This function should be called in the train loop when the accumulation 
        is done to randomly choose a new object value.
        """
        assert self.learnable_mode == 3
        self.current_object_idx = np.random.choice(len(
            self.train_data_subsets))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = i % self.num_images

        if self.learnable_mode != 3:
            image_paths = self.image_paths
            placeholder_object_token = self.placeholder_object_tokens[0]
            assert len(self.placeholder_object_tokens) == 1

        else:
            np.random.seed(int(time.time() * 1000) % 2**32)
            self.current_object_idx = np.random.choice(
                len(self.train_data_subsets))
            current_object = str(self.current_object_idx)

            image_paths = self.image_paths[current_object]
            placeholder_object_token = self.lookup_object_to_placeholder_object_token[
                current_object]
            idx = i % len(image_paths)

        image_path = image_paths[idx]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example = dict()
        example['image_idx'] = idx

        template = random.choice(self.templates)

        # different modes build the caption differently. They also supply the
        # list of placeholder token ids
        if self.learnable_mode == 0:
            example['text'] = template.format(placeholder_object_token)
            example['input_ids_placeholder_view'] = torch.tensor(-1)
            example['input_ids_placeholder_object'] = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(placeholder_object_token))

        elif self.learnable_mode in (1, 2, 3, 4, 5):
            if self.camera_representation == "spherical":
                # todo, make it a lookup table instead of this hack
                view_token = f"<view_{image_path.stem.split('___')[-1]}>"

            elif self.camera_representation == "dtu-12d":
                cam_key, _ = TextualInversionDataset.dtu_cam_info_from_fname(
                    image_path)
                view_token = self.lookup_camidx_to_view_token[cam_key]

            else:
                raise NotImplementedError()
            assert view_token in self.placeholder_view_tokens

            if self.learnable_mode == 1:
                if self.caption_strategy == 0:
                    text = f"{view_token}. A photo of a {self.fixed_object_token}"
                elif self.caption_strategy == 1:
                    text = f"A photo of a {self.fixed_object_token} in the stye of {view_token}"
                elif self.caption_strategy == 2:
                    text = f"A photo of a {self.fixed_object_token} {view_token}"
                else:
                    raise NotImplementedError()
                if self.fixed_object_token_pretrained:
                    example['input_ids_placeholder_object'] = torch.tensor(
                        self.tokenizer.convert_tokens_to_ids(
                            placeholder_object_token))
                else:
                    example['input_ids_placeholder_object'] = torch.tensor(-1)

            elif self.learnable_mode in (2, 3, 4, 5):
                text = f"{view_token}. A photo of a {placeholder_object_token}"
                example['input_ids_placeholder_object'] = torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(
                        placeholder_object_token))

            example['text'] = text
            example['input_ids_placeholder_view'] = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(view_token))

        else:
            raise NotImplementedError()

        example["input_ids"] = self.tokenizer(
            example['text'],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)

        if "dtu" in str(self.data_root):
            # first setting: bottom-pad with black and resize to 512
            if self.dtu_preprocess_key == 0:
                # black padding at the bottom of the image
                padding = (0, 0, 0, 400)
                image = ImageOps.expand(image, padding, fill='black')
                assert image.size == (1600, 1600)
                image = image.resize((512, 512), resample=self.interpolation)

            elif self.dtu_preprocess_key == 1:
                image = image.resize((512, 384), resample=self.interpolation)

            elif self.dtu_preprocess_key == 2:
                image = image.resize((768, 576), resample=self.interpolation)
            else:
                raise NotImplementedError()

        elif 'llff' in str(self.data_root):
            pass  # no resizing

        # non dtu-datasets use the original options
        else:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)

        img_size = image.size
        if self.learnable_mode == 0:
            image = self.flip_transform(image)
        if self.augmentation_key > 0:
            image = self.augmentations(image)
            assert image.size == img_size

        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        return example
