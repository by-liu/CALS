import os
import sys
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from calibrate.utils import setup_logger


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    setup_logger(cfg.work_dir, job_name=cfg.job_name)
    logger.info("Launches command : {}".format(" ".join(sys.argv)))

    trainer = instantiate(cfg.train.object, cfg)
    trainer.run()

    logger.info("Job complete !\n")


if __name__ == "__main__":
    main()
