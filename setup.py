from setuptools import setup

setup(
    name="vapo_aff",
    version="1.0",
    description="Python Distribution Utilities",
    packages=["vapo_aff"],
    install_requires=[
        "torch",
        "pytorch-lightning",
        "segmentation-models-pytorch",
        "hydra-core",
        "opencv-python",
        "matplotlib",
        "pypng",
        "scipy",
        "omegaconf",
        "gym",
        "wandb",
        "pybullet",
        "scikit-learn",
    ],
)
