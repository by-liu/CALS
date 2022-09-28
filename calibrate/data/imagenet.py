import os.path as osp
from typing import Tuple, Any, Optional
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

from timm.data import create_transform

from ..utils.file_io import load_list


class MyImagenet(ImageNet):
    def __init__(self, root: str, use_mysplit: bool = False, return_ind: bool = False, split: str = 'train', **kwargs: Any) -> None:
        super().__init__(root, "val" if split == "test" else split, **kwargs)
        self.return_ind = return_ind
        if use_mysplit:
            self.filter(split)

    def filter(self, split) -> None:
        """We split out a subset from the original validataion set as val set and use the rest for testing."""
        if split == "val":
            val_ind = load_list(osp.join(self.root, "val_ind.txt"))
            val_ind = set([int(i) for i in val_ind])
            self.samples = [s for i, s in enumerate(self.samples) if i in val_ind]
        elif split == "test":
            test_ind = load_list(osp.join(self.root, "test_ind.txt"))
            test_ind = set([int(i) for i in test_ind])
            self.samples = [s for i, s in enumerate(self.samples) if i in test_ind]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        if self.return_ind:
            return img, target, index
        else:
            return img, target


def build_transform(is_train: bool = True, input_size: int = 224, **kwargs: Any) -> transforms.Compose:
    transform = create_transform(
        input_size=input_size,
        is_training=is_train,
        color_jitter=0.4,
        re_prob=0.25,  # random erase prob
        re_mode="pixel",  # random erase mode
        re_count=1,  # random erase count
        interpolation="bicubic",
    )

    return transform


def build_train_val_dataset(data_root, input_size=224, use_mysplit=False):
    train_dataset = MyImagenet(
        data_root,
        split="train",
        transform=build_transform(is_train=True, input_size=input_size),
        use_mysplit=use_mysplit,
    )

    val_dataset = MyImagenet(
        data_root,
        split="val",
        transform=build_transform(is_train=False, input_size=input_size),
        use_mysplit=use_mysplit,
    )

    return train_dataset, val_dataset


def build_test_dataset(data_root, input_size=224, use_mysplit=False):
    test_dataset = MyImagenet(
        data_root,
        split="test",
        transform=build_transform(is_train=False, input_size=input_size),
        use_mysplit=use_mysplit,
    )

    return test_dataset
