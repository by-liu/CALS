<!-- omit in toc -->
# Class Adaptive Network Calibration

[Bingyuan Liu](https://by-liu.github.io/), [Jérôme Rony](https://scholar.google.ca/citations?user=5_jpGD0AAAAJ&hl=en), [Adrian Galdran](https://scholar.google.es/citations?user=VKx-rswAAAAJ&hl=es), [Jose Dolz](https://josedolz.github.io/), [Ismail Ben Ayed](https://profs.etsmtl.ca/ibenayed/)

```To Appear at CVPR 2023```

[[`arXiv`](https://arxiv.org/abs/2211.15088)][[`BibTeX`](#CitingCALS)] 


## Install:

The pytorch version we used:
```
torch==1.12.1
torchvision==0.13.1
```

setup:
```
pip install -e .
```

Install window_process kernal for Swin-T model(Optional)
```
cd kernels/window_process
pip install -e .
```

## Data preparation

For the datasets (i.e., Tiny-ImageNet, ImageNet, ImageNetLT, and VOC2012), please refer to their official citer for download.
We provide the data splits we used under [data_splits](data_splits/). Please put them to the root directory (or VOC2012/ImageSets/Segmentation for VOC2012) of your dataset.

**Important Note** : Before you run the code, please add the absolute path of the root directory for the related data configs in [configs/data](configs/data/), or pass it in the running commands.

## Usage:

### Arguments

<details><summary>python tools/train_net.py --help</summary>
<p>

```python
dist_train is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

data: imagenet, imagenet_lt, tiny_imagenet, voc2012
dist: pytorch, slurm
lag: aug_lagrangian, aug_lagrangian_class
loss: ce, cpc, focal, focal_adaptive, logit_margin, logit_margin_plus, ls, mmce, penalty_ent, soft_ce
lr_scheduler: cosine, multi_step, one_cycle, plateau, step
mixup: mixup
model: SwinV2-T, deeplabv3, resnet, resnet101_tiny, resnet50_tiny
optim: adam, adamw, sgd
test: local, segment
train: dist, dist_lag, dist_plus, lag_segment, local, segment
wandb: my


== Config ==
Override anything in the config (foo.bar=value)

data:
  name: imagenet
  data_root: YOUR_DATA_ROOT
  input_size: 224
  train_batch_size: 256
  val_batch_size: 250
  test_batch_size: 250
  return_ind: false
  use_mysplit: true
  num_workers: 8
  pin_memory: true
  object:
    trainval:
      _target_: calibrate.data.imagenet.build_train_val_dataset
      data_root: ${data.data_root}
      input_size: ${data.input_size}
      use_mysplit: ${data.use_mysplit}
    test:
      _target_: calibrate.data.imagenet.build_test_dataset
      data_root: ${data.data_root}
      input_size: ${data.input_size}
      use_mysplit: ${data.use_mysplit}
model:
  name: resnet50
  num_classes: 10
  pretrained: false
  drop_rate: 0.0
  object:
    _target_: timm.create_model
    model_name: ${model.name}
    pretrained: ${model.pretrained}
    num_classes: ${model.num_classes}
    drop_rate: ${model.drop_rate}
loss:
  name: ce
  ignore_index: -100
  object:
    _target_: torch.nn.CrossEntropyLoss
    ignore_index: ${loss.ignore_index}
    reduction: mean
optim:
  name: adamw
  lr: 0.0005
  weight_decay: 0.05
  object:
    _target_: torch.optim.AdamW
    lr: ${optim.lr}
    betas:
    - 0.9
    - 0.999
    weight_decay: ${optim.weight_decay}
    eps: 1.0e-08
lr_scheduler:
  name: cosine
  min_lr: 5.0e-06
  warmup_lr: 5.0e-07
  warmup_epochs: 1
  cycle_decay: 0.1
  object:
    _target_: timm.scheduler.cosine_lr.CosineLRScheduler
    t_initial: ${train.max_epoch}
    lr_min: ${lr_scheduler.min_lr}
    cycle_mul: 1
    cycle_decay: ${lr_scheduler.cycle_decay}
    warmup_lr_init: ${lr_scheduler.warmup_lr}
    warmup_t: ${lr_scheduler.warmup_epochs}
    cycle_limit: 1
    t_in_epochs: true
train:
  name: dist
  max_epoch: 200
  clip_grad: 2.0
  clip_mode: norm
  resume: true
  keep_checkpoint_num: 1
  keep_checkpoint_interval: 0
  use_amp: true
  evaluate_logits: true
  object:
    _target_: calibrate.engine.DistributedTrainer
mixup:
  name: mixup
  enable: false
  mixup_alpha: 0.4
  mode: pair
  label_smoothing: 0
  object:
    _target_: timm.data.mixup.Mixup
    mixup_alpha: ${mixup.mixup_alpha}
    mode: ${mixup.mode}
    label_smoothing: ${mixup.label_smoothing}
    num_classes: ${model.num_classes}
dist:
  launch: python
  backend: nccl
wandb:
  enable: false
  project: NA
  entity: NA
  tags: train
test:
  name: local
  checkpoint: best.pth
  save_logits: false
  object:
    _target_: calibrate.engine.Tester
job_name: ${hydra:job.name}
work_dir: ${hydra:run.dir}
seed: 1
log_period: 50
calibrate:
  num_bins: 15
  visualize: false
post_temperature:
  enable: false
  learn: false
  grid_search_interval: 0.1
  cross_validate: ece


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```
</p>
</details>

### Example


#### TinyImageNet

Ours
```python
OMP_NUM_THREADS=8 torchrun \
    --master_port 29500 --nnodes=1 --nproc_per_node=1 \
     tools/dist_train.py \
     dist=pytorch model=resnet50_tiny train.max_epoch=100 \
     data=tiny_imagenet data.train_batch_size=128 \
     optim=sgd optim.lr=0.1 lr_scheduler=multi_step \
     train=dist_lag loss=ce +lag=aug_lagrangian_class lag.margin=10
```

Baselines:

CE (Cross Entropy)
```python
OMP_NUM_THREADS=8 torchrun \
    --master_port 29500 --nnodes=1 --nproc_per_node=1 \
     tools/dist_train.py \
     dist=pytorch model=resnet50_tiny train=dist train.max_epoch=100 \
     data=tiny_imagenet data.train_batch_size=128 \
     optim=sgd optim.lr=0.1 lr_scheduler=multi_step \
     train=dist loss=ce
```

MbLS (Margin based Label Smoothing)
```python
OMP_NUM_THREADS=8 torchrun \
    --master_port 29500 --nnodes=1 --nproc_per_node=1 \
     tools/dist_train.py \
     dist=pytorch model=resnet50_tiny train=dist train.max_epoch=100 \
     data=tiny_imagenet data.train_batch_size=128 \
     optim=sgd optim.lr=0.1 lr_scheduler=multi_step \
     train=dist loss=logit_margin
```


#### ImageNetLT:

Ours:
```python
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --master_port 29500 --nnodes=1 --nproc_per_node=2 \
    tools/dist_train.py \
    model=resnet model.num_classes=1000 \
    data=imagenet_lt data.train_batch_size=256 \
    optim.lr=0.0005 lr_scheduler.min_lr=5e-6 lr_scheduler.warmup_lr=5e-7 \
    train=dist_lag train.max_epoch=200 \
    loss=ce +lag=aug_lagrangian_class lag.margin=10 lag.lambd_step=1
```

Baselines:

CE (Cross Entropy):
```python
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --master_port 29500 --nnodes=1 --nproc_per_node=2 \
    tools/dist_train.py \
    model=resnet model.num_classes=1000 \
    data=imagenet_lt data.train_batch_size=256 \
    optim.lr=0.0005 lr_scheduler.min_lr=5e-6 lr_scheduler.warmup_lr=5e-7 \
    train=dist train.max_epoch=200 \
    loss=ce
```

MbLS (Margin based Label Smoothing):
```python
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --master_port 29500 --nnodes=1 --nproc_per_node=2 \
    tools/dist_train.py \
    model=resnet model.num_classes=1000 \
    data=imagenet_lt data.train_batch_size=256 \
    optim.lr=0.0005 lr_scheduler.min_lr=5e-6 lr_scheduler.warmup_lr=5e-7 \
    train=dist train.max_epoch=200 \
    loss=logit_margin
```


<!-- omit in toc -->
## <a name="CitingCALS"></a>Citing CALS


```BibTeX
@article{liu2022cals,
  title = {Class Adaptive Network Calibration}, 
  author={Bingyuan Liu and Jérôme Rony and Adrian Galdran and Jose Dolz and Ismail Ben Ayed},
  journal = {arXiv preprint arXiv:2211.15088},
  year={2022},
}
```
