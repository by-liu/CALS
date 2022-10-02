"""
Create train, val, test iterators for Tiny ImageNet.
Train set size: 100000
Val set size: 10000
Test set size: 10000
Number of classes: 200
Link: https://tiny-imagenet.herokuapp.com/
"""

import os
import torch
import numpy as np
import glob
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ..utils.file_io import load_list


EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'


class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False, return_ind=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        self.return_ind = return_ind

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def filter(self, mysplit="train") -> None:
        """apply our split with train-val-test setting"""
        if mysplit == 'train':
            train_ind = load_list(os.path.join(self.root, 'train_ind.txt'))
            train_ind = set([int(i) for i in train_ind])
            self.image_paths = [s for i, s in enumerate(self.image_paths) if i in train_ind]
        elif mysplit == 'val':
            val_ind = load_list(os.path.join(self.root, 'val_ind.txt'))
            val_ind = set([int(i) for i in val_ind])
            self.image_paths = [s for i, s in enumerate(self.image_paths) if i in val_ind]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            if self.return_ind:
                return img, self.labels[os.path.basename(file_path)], index
            else:
                return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        if (img.mode == 'L'):
            img = img.convert('RGB')
        return self.transform(img) if self.transform else img


class TinyImageNetC(TinyImageNet):
    def __init__(
        self, root, root_c,
        transform=None,
        target_transform=None,
        corruption="gaussian_noise",
        severity=5,
    ):
        super().__init__(
            root,
            split="test",
            transform=transform,
            target_transform=target_transform,
            in_memory=False
        )

        self.root_c = os.path.expanduser(root_c)
        self.corruption = corruption
        self.severity = str(severity)

        self.load_images()

    def load_images(self):
        self.image_dir = os.path.join(
            self.root_c, self.corruption, self.severity
        )
        self.image_paths = []
        self.labels = []
        classes = sorted(os.listdir(self.image_dir))
        for cls in classes:
            cls_img_dir = os.path.join(self.image_dir, cls)
            label = self.label_text_to_number[cls]
            for name in sorted(os.listdir(cls_img_dir)):
                self.image_paths.append(os.path.join(cls_img_dir, name))
                self.labels.append(label)

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        label = self.labels[index]

        img = self.read_image(file_path)

        return img, label


def get_data_loader(root,
                    batch_size,
                    split='train',
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the Tiny Imagenet dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - root: The root directory for TinyImagenet dataset
    - batch_size: how many samples per batch to load.
    - split: Can be train/val/test. For train we apply the data augmentation techniques.
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
    ])

    train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])

    # load the dataset
    data_dir = root

    if (split == 'train'):
        dataset = TinyImageNet(data_dir,
                               split='train',
                               transform=train_transform,
                               in_memory=True)
    else:
        dataset = TinyImageNet(data_dir,
                               split='val',
                               transform=val_test_transform,
                               in_memory=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=True
    )

    return data_loader


def build_transform(is_train: bool = True, for_vit: bool = False):
    if for_vit:
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        if is_train:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            return train_transform
        else:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize
            ])
            return val_transform
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if is_train:
            train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            return train_transform
        else:
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
            return val_transform


def build_train_val_dataset(root, for_vit=False):
    train_dataset = TinyImageNet(root,
                                 split="train",
                                 transform=build_transform(is_train=True, for_vit=for_vit),
                                 in_memory=False)
    train_dataset.filter(mysplit="train")

    val_dataset = TinyImageNet(root,
                               split="train",
                               transform=build_transform(is_train=False, for_vit=for_vit),
                               in_memory=False)
    val_dataset.filter(mysplit="val")

    return train_dataset, val_dataset


def build_test_dataset(root, for_vit=False):
    test_dataset = TinyImageNet(root,
                                split="val",
                                transform=build_transform(is_train=False, for_vit=for_vit),
                                in_memory=False)
    return test_dataset


def get_train_val_loader(root,
                         batch_size,
                         val_samples_per_class=50,
                         random_seed=1,
                         shuffle=True,
                         num_workers=4,
                         pin_memory=False,
                         for_vit=False,
                         return_ind=False):
    np.random.seed(random_seed)

    if for_vit:
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])

        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = TinyImageNet(root,
                                 split="train",
                                 transform=train_transform,
                                 in_memory=False,
                                 return_ind=return_ind)
    val_dataset = TinyImageNet(root,
                               split="train",
                               transform=val_transform,
                               in_memory=False,
                               return_ind=return_ind)

    num_train = len(train_dataset)
    class_indices = {}
    for i in range(num_train):
        file_path = train_dataset.image_paths[i]
        label = train_dataset.labels[os.path.basename(file_path)]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)

    train_indices, val_indices = [], []
    for label in class_indices:
        indices = class_indices[label]
        if shuffle:
            np.random.shuffle(indices)
            train_indices.extend(indices[:-val_samples_per_class])
            val_indices.extend(indices[-val_samples_per_class:])

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader


def get_test_loader(root,
                    batch_size,
                    num_workers=4,
                    pin_memory=False,
                    for_vit=False):

    if for_vit:
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    dataset = TinyImageNet(root, split="val",
                           transform=test_transform,
                           in_memory=False)

    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader


def get_test_c_loader(root, root_c, batch_size,
                      corruption="gaussian_noise",
                      severity=5,
                      num_workers=4,
                      pin_memory=False,
                      for_vit=False):

    if for_vit:
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    dataset = TinyImageNetC(root, root_c,
                            transform=test_transform,
                            corruption=corruption,
                            severity=severity)

    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader