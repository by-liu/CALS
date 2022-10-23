import os
import sys
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist

from calibrate.utils import setup_logger

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    setup_logger(cfg.work_dir, job_name=cfg.job_name)

    tester = instantiate(cfg.test.object, cfg)
    tester.run()

    logger.info("Job complete !\n")


if __name__ == "__main__":
    main()
