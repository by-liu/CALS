from setuptools import setup, find_packages

setup(
    name="calibrate",
    version="0.2",
    author="",
    description="For awesome calibration research",
    packages=find_packages(),
    python_requries=">=3.9",
    install_requires=[
        # Please install torch and torchvision libraries before running this script
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "ipdb==0.13.9",
        "albumentations==1.2.1",
        # "opencv-python==4.5.1.48",
        "hydra-core==1.2.0",
        "flake8==4.0.1",
        "wandb==0.13.1",
        "terminaltables==3.1.10",
        "matplotlib==3.5.3",
        "plotly==5.10.0",
        "timm==0.4.12",
        # "segmentation_models_pytorch==0.2.1",
    ],
)
