# Derived from VREnv/vr_env/camera/static_camera.py


import numpy as np
import pybullet as p
from vapo.agent.camera import Camera

class StaticCamera(Camera):
    def __init__(
        self,
        fov,
        aspect,
        nearval,
        farval,
        width,
        height,
        look_at,
        look_from,
        name,
        up_vector=[0, 0, 1],
        robot_id=None,
        objects=None,
    ):
        """
        Initialize the camera
        Args:
            argument_group: initialize the camera and add needed arguments to argparse

        Returns:
            None
        """
        self.nearval = nearval
        self.farval = farval
        self.fov = fov
        self.aspect = aspect
        self.look_from = look_from
        self.look_at = look_at
        self.up_vector = up_vector
        self.width = width
        self.height = height
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=self.nearval, farVal=self.farval
        )
        self.name = name
