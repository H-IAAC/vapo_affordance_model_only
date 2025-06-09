# Derived from VREnv/vr_env/envs/play_table_env.py

import logging
import os
from pathlib import Path
import pickle
import pkgutil
import re
import sys
import time

import cv2
import gym
import gym.utils
import gym.utils.seeding
import hydra
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

# A logger for this file
log = logging.getLogger(__name__)


class PlayTableSimEnv(gym.Env):
    def __init__(
        self,
        cameras,
        **kwargs,
    ):
        self.p = p

        # Load Env
        self.load()

        # init cameras after scene is loaded to have robot id available
        self.cameras = [
            hydra.utils.instantiate(
                cameras[name], cid=self.cid, robot_id=self.robot.robot_uid, objects=self.scene.get_objects()
            )
            for name in cameras
        ]

    def __del__(self):
        self.close()

    def get_camera_obs(self):
        assert self.cameras is not None
        rgb_obs = {}
        depth_obs = {}
        for cam in self.cameras:
            rgb, depth = cam.render()
            rgb_obs[f"rgb_{cam.name}"] = rgb
            depth_obs[f"depth_{cam.name}"] = depth
        return rgb_obs, depth_obs

    def get_obs(self):
        """Collect camera, robot and scene observations."""
        rgb_obs, depth_obs = self.get_camera_obs()
        obs = {"rgb_obs": rgb_obs, "depth_obs": depth_obs}
        obs.update(self.get_state_obs())
        return obs

    def get_state_obs(self):
        """
        Collect state observation dict
        --state_obs
            --robot_obs
                --robot_state_full
                    -- [tcp_pos, tcp_orn, gripper_opening_width]
                --gripper_opening_width
                --arm_joint_states
                --gripper_action}
            --scene_obs
        """
        robot_obs, robot_info = self.robot.get_observation()
        scene_obs = self.scene.get_obs()
        obs = {"robot_obs": robot_obs, "scene_obs": scene_obs}
        return obs


def get_env(dataset_path, obs_space=None, **kwargs):
    from pathlib import Path

    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(Path(dataset_path) / ".hydra" / "merged_config.yaml")

    if obs_space is not None:
        exclude_keys = set(render_conf.cameras.keys()) - {
            re.split("_", key)[1] for key in obs_space["rgb_obs"] + obs_space["depth_obs"]
        }
        for k in exclude_keys:
            del render_conf.cameras[k]
    
    env = hydra.utils.instantiate(render_conf.env, show_gui=False, use_vr=False, use_scene_info=False)
    return env
