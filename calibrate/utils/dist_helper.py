import os
import os.path as osp
import socket
import subprocess
from typing import Callable, List, Optional, Tuple

import torch
from torch import distributed as dist
import logging
import sys


def _find_free_port() -> str:
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _is_free_port(port: int) -> bool:
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def init_dist_pytorch(backend: str = "nccl") -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend, init_method="env://", world_size=world_size, rank=rank
    )


def init_dist_slurm(backend: str = "nccl", port: Optional[int] = None) -> None:
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend, rank=proc_id, world_size=ntasks)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def gather(tensor, tensor_list=None, root=0):
    rank = dist.get_rank()
    if rank == root:
        assert tensor_list is not None
        dist.gather(tensor, gather_list=tensor_list)
    else:
        dist.gather(tensor, dst=root)


def build_dist_data_loader(
    dataset,
    batch_size,
    world_size=1,
    rank=0,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


def setup_dist_logger(save_dir, dist_rank, job_name="train", level=logging.INFO):
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(levelname)s][%(filename)s:%(lineno)s - %(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    # create console handler for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # create file handler for all processes
    file_handler = logging.FileHandler(
        osp.join(save_dir, f"{job_name}_rank{dist_rank}.log"),
        mode="a",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger at rank{dist_rank} is set up.")


def setup_logger(save_dir, job_name="train", level=logging.INFO):
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(levelname)s][%(filename)s:%(lineno)s - %(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    # create console handler for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create file handler for all processes
    file_handler = logging.FileHandler(
        osp.join(save_dir, f"{job_name}.log"),
        mode="a",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logger is set up.")
