import ipdb
import sys
import os 

import pyrallis
from diffusers.utils import check_min_version

sys.path.append(".")
sys.path.append("..")

from training.coach import Coach
from training.config import RunConfig
from utils.fixseed import fixseed 

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")


@pyrallis.wrap()
def main(cfg: RunConfig):
    fixseed(cfg.seed)
    prepare_directories(cfg=cfg)
    coach = Coach(cfg)
    coach.train()


def prepare_directories(cfg: RunConfig):
    cfg.log.exp_dir = cfg.log.exp_dir / cfg.log.exp_name
    if os.path.exists(cfg.log.exp_dir) and not cfg.log.overwrite_ok:
        raise ValueError(f"Experiment folder already exists and overwrite_ok=False: [{cfg.log.exp_dir}]")
    cfg.log.exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.log.logging_dir = cfg.log.exp_dir / cfg.log.logging_dir
    cfg.log.logging_dir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main()
