import glob
import os

import hydra
import numpy as np
from omegaconf import OmegaConf
from vapo.affordance.affordance_model import AffordanceModel


def get_files_regex(path, search_str, recursive):
    files = glob.glob(os.path.join(path, search_str), recursive=recursive)
    if not files:
        print("No *.%s files found in %s" % (search_str, path))
    files.sort()
    return files


# Ger valid numpy files with raw data
def get_files(path, extension, recursive=False):
    if not os.path.isdir(path):
        print("path does not exist: %s" % path)
    search_str = "*.%s" % extension if not recursive else "**/*.%s" % extension
    files = get_files_regex(path, search_str, recursive)
    return files


def get_abs_path(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def torch_to_numpy(x):
    return x.detach().cpu().numpy()


# Load affordance model from function parameters. For a general,
# hydra dependant functionality, check function load_from_hydra()
# in vapo/affordance/utils/utils.py
def init_aff_net(affordance_cfg, cam_str=None, in_channels=1):
    aff_net = None
    if affordance_cfg is not None:
        if cam_str is not None:
            aff_cfg = affordance_cfg["%s_cam" % cam_str]
        else:
            aff_cfg = affordance_cfg
        if "use" in aff_cfg and aff_cfg.use:
            path = aff_cfg.model_path
            path = get_abs_path(path)
            # Configuration of the model
            hp = {
                "cfg": aff_cfg.hyperparameters.cfg,
                "n_classes": aff_cfg.hyperparameters.n_classes,
                "input_channels": in_channels,
            }
            hp = OmegaConf.create(hp)
            # Create model
            if os.path.exists(path):
                aff_net = AffordanceModel.load_from_checkpoint(path, **hp)
                aff_net.cuda()
                aff_net.eval()
                print("obs_wrapper: %s cam affordance model loaded" % cam_str)
            else:
                # affordance_cfg = None
                raise TypeError("Path does not exist: %s" % path)
    return aff_net
