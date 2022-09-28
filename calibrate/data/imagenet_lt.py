import os.path as osp
from typing import Tuple, Any, Optional
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

from timm.data import create_transform


class ImagenetLT(Dataset):
    def __init__(
        self, root: str, split: str = "train",
        transform: Optional[transforms.Compose] = None
    ) -> None:
        self.root = root
        self.split = split
        self.transform = transform
        self.load_list()

    def load_list(self):
        self.img_path = []
        self.labels = []
        path = osp.join(self.root, f"ImageNet_LT_{self.split}.txt")
        with open(path, "r") as f:
            for line in f:
                img, label = line.strip().split()
                self.img_path.append(osp.join(self.root, img))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        path = self.img_path[index]
        label = self.labels[index]

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __repr__(self) -> str:
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


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


def build_train_val_dataset(data_root, input_size=224):
    train_dataset = ImagenetLT(
        data_root,
        split="train",
        transform=build_transform(is_train=True, input_size=input_size),
    )

    val_dataset = ImagenetLT(
        data_root,
        split="val",
        transform=build_transform(is_train=False, input_size=input_size),
    )

    return train_dataset, val_dataset


def build_test_dataset(data_root, input_size=224):
    test_dataset = ImagenetLT(
        data_root,
        split="test",
        transform=build_transform(is_train=False, input_size=input_size),
    )

    return test_dataset
