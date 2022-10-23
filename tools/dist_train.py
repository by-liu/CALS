import os
import sys
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist

from calibrate.utils import setup_dist_logger, init_dist_pytorch, init_dist_slurm


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    if cfg.dist.launch == "python":
        init_dist_pytorch(cfg.dist.backend)
    elif cfg.dist.launch == "slurm":
        init_dist_slurm(cfg.dist.backend, cfg.dist.port)
    else:
        raise NotImplementedError(f"Unsupported launch method: {cfg.dist.launch}")
    dist.barrier()

    setup_dist_logger(cfg.work_dir, dist.get_rank(), job_name=cfg.job_name)
    logger.info(f"Distributed initialized : rank - {dist.get_rank()}, world_size - {dist.get_world_size()}")

    logger.info("Rank {} launches command : {}".format(dist.get_rank(), " ".join(sys.argv)))

    trainer = instantiate(cfg.train.object, cfg)
    trainer.run()

    logger.info("Job complete !\n")


if __name__ == "__main__":
    main()
