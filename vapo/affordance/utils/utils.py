import os

import hydra
from omegaconf import OmegaConf
from scipy.spatial.transform.rotation import Rotation as R
from torchvision import transforms

from vapo.affordance.affordance_model import AffordanceModel


def get_abs_path(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def euler_to_quat(euler_angles):
    """xyz euler angles to xyzw quat"""
    return R.from_euler("xyz", euler_angles).as_quat()


def quat_to_euler(quat):
    """xyz euler angles to xyzw quat"""
    return R.from_quat(quat).as_euler("xyz")


def get_transforms(transforms_cfg, img_size=None):
    transforms_lst = []
    transforms_config = transforms_cfg.copy()
    for cfg in transforms_config:
        if ("size" in cfg) and img_size is not None:
            cfg.size = img_size
        transforms_lst.append(hydra.utils.instantiate(cfg))

    return transforms.Compose(transforms_lst)


# Load affordance model from a hydra configuration. Hyperparameters
# and model configuration are loaded from the hydra config file.
# Has the same functionality of load_cfg() and init_aff_net()
# in vapo/utils/utils.py
def load_from_hydra(cfg_path, cfg):
    if os.path.exists(cfg_path):
        run_cfg = OmegaConf.load(cfg_path)
    else:
        print("utils.py: Path does not exist %s" % cfg_path)
        run_cfg = cfg
    
    model_cfg = run_cfg.model_cfg
    model_cfg.hough_voting = cfg.model_cfg.hough_voting
    model_path = get_abs_path(run_cfg.model_path)

    # Load model
    if os.path.isfile(model_path):
        model = AffordanceModel.load_from_checkpoint(model_path, cfg=model_cfg)
        model.cuda()
        model.eval()
        print("Model loaded")
    else:
        model = None
        raise TypeError("No file found in: %s " % model_path)
    return model, run_cfg


def torch_to_numpy(x):
    return x.detach().cpu().numpy()
