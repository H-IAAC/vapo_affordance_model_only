import logging
import math
import os

import cv2
import gym
from gym import spaces
import numpy as np
import pybullet as p
import torch

from env_utils import EglDeviceNotFoundError, get_egl_device_id
from play_table_env import PlayTableSimEnv
from vapo.utils.utils import get_3D_end_points
from vapo.wrappers.utils import find_cam_ids

logger = logging.getLogger(__name__)


class PlayTableRL(PlayTableSimEnv):
    def __init__(self, task="slide", sparse_reward=False, max_counts=50, viz=False, save_images=False, **args):
        if "use_egl" in args and args["use_egl"]:
            # if("CUDA_VISIBLE_DEVICES" in os.environ):
            #     device_id = os.environ["CUDA_VISIBLE_DEVICES"]
            #     device = int(device_id)
            # else:
            device = torch.cuda.current_device()
            device = torch.device(device)
            self.set_egl_device(device)
        super(PlayTableRL, self).__init__(**args)
        self.task = task
        _action_space = np.ones(7)
        self.action_space = spaces.Box(_action_space * -1, _action_space)
        obs_space_dict = {
            "scene_obs": gym.spaces.Box(low=0, high=1.5, shape=(3,)),
            "robot_obs": gym.spaces.Box(low=-0.5, high=0.5, shape=(7,)),
            "rgb_obs": gym.spaces.Box(low=0, high=255, shape=(3, 300, 300)),
            "depth_obs": gym.spaces.Box(low=0, high=255, shape=(1, 300, 300)),
        }
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        self.sparse_reward = sparse_reward
        self.offset = np.array([*args["offset"], 1])
        self.reward_fail = args["reward_fail"]
        self.reward_success = args["reward_success"]
        self._obs_it = 0
        self.viz = viz
        self.save_images = save_images
        self.cam_ids = find_cam_ids(self.cameras)

        self._rand_scene = "rand_scene" in args
        _initial_obs = self.get_obs()["robot_obs"]
        self._start_orn = _initial_obs[3:6]

        self.load()

        self._target = task
        # x1,y1,z1, width, height, depth (x,y,z) in meters]
        self.box_pos = self.scene.object_cfg["fixed_objects"]["bin"]["initial_pos"]
        w, h, d = 0.24, 0.4, 0.08
        self.box_3D_end_points = get_3D_end_points(*self.box_pos, w, h, d)

    @property
    def obs_it(self):
        return self._obs_it

    @property
    def target(self):
        return self._target

    @obs_it.setter
    def obs_it(self, value):
        self._obs_it = value

    @target.setter
    def target(self, value):
        self._target = value
        self.scene.target = value

    def set_egl_device(self, device):
        assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
        cuda_id = device.index if device.type == "cuda" else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to VREnv README"
            )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def get_target_pos(self):
        if self.task == "slide":
            link_id = self.scene.get_info()["fixed_objects"]["table"]["uid"]
            targetWorldPos = self.p.getLinkState(link_id, 2, physicsClientId=self.cid)[0]
            targetState = self.p.getJointState(link_id, 2, physicsClientId=self.cid)[0]

            # only keep x dim
            targetWorldPos = [targetWorldPos[0] - 0.1, 0.75, 0.74]
            targetState = self._normalize(targetState, 0, 0.56)
        elif self.task == "hinge":
            link_id = self.scene.get_info()["fixed_objects"]["hinged_drawer"]["uid"]
            targetWorldPos = self.p.getLinkState(link_id, 1, physicsClientId=self.cid)[0]
            # targetState = self.p.getJointState(link_id, 1, physicsClientId=self.cid)[0]

            targetWorldPos = [targetWorldPos[0] + 0.02, targetWorldPos[1], 1]
            # table is id 0,
            # hinge door state(0) increases as it moves to left 0 to 1.74
            targetState = self.p.getJointState(0, 0, physicsClientId=self.cid)[0]
            targetState = self._normalize(targetState, 0, 1.74)
        elif self.task == "drawer":  # self.task == "drawer":
            link_id = self.scene.get_info()["fixed_objects"][self.task]["uid"]
            targetWorldPos = self.p.getLinkState(link_id, 0, physicsClientId=self.cid)[0]
            targetState = self.p.getJointState(link_id, 0, physicsClientId=self.cid)[0]
            targetWorldPos = [-0.05, targetWorldPos[1] - 0.41, 0.53]
            # self.p.addUserDebugText("O", textPosition=targetWorldPos, textColorRGB=[0, 0, 1])
            targetState = self._normalize(targetState, 0, 0.23)
        else:
            lifted = False
            for name in self.scene.table_objs:
                target_obj = self.scene.get_info()["movable_objects"][name]
                base_pos = p.getBasePositionAndOrientation(target_obj["uid"], physicsClientId=self.cid)[0]
                # if(p.getNumJoints(target_obj["uid"]) == 0):
                #     pos = base_pos
                # else:
                #     pos = p.getLinkState(target_obj["uid"], 0)[0]

                # self.p.addUserDebugText("O", textPosition=pos,
                #                         textColorRGB=[0, 0, 1])
                # 2.5cm above initial position and object not already in box
                if base_pos[-1] >= target_obj["initial_pos"][-1] + 0.020 and not self.obj_in_box(name):
                    lifted = True
            targetState = lifted
            # Return position of current target for training
            curr_target_uid = self.scene.get_info()["movable_objects"][self.target]["uid"]
            if p.getNumJoints(curr_target_uid) == 0:
                targetWorldPos = p.getBasePositionAndOrientation(curr_target_uid, physicsClientId=self.cid)[0]
            else:
                targetWorldPos = p.getLinkState(curr_target_uid, 0)[0]
        return targetWorldPos, targetState  # normalized

    def save_and_viz_obs(self, obs):
        if self.viz:
            for cam_name, _ in self.cam_ids.items():
                if ("gripper_aff" not in self.observation_space.spaces
                    or cam_name=="render" or cam_name == "static"):
                    cv2.imshow("%s_cam" % cam_name, obs["rgb_obs"]["rgb_%s" % cam_name][:, :, ::-1])
            cv2.waitKey(1)
        if self.save_images:
            for cam_name, _ in self.cam_ids.items():
                os.makedirs("./images/%s_orig" % cam_name, exist_ok=True)
                cv2.imwrite(
                    "./images/%s_orig/img_%04d.png" % (cam_name, self.obs_it),
                    obs["rgb_obs"]["rgb_%s" % cam_name][:, :, ::-1],
                )
        self.obs_it += 1
