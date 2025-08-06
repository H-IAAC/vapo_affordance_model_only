# Derived from VREnv/vr_env/camera/camera.py

import math
import numpy as np

class Camera:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def project(self, point):
        """
        Projects a world point in homogeneous coordinates to pixel coordinates
        Args
            point: np.array of len 4; indicates the desired point to project
        Output
            (x, y): tuple (u, v); pixel coordinates of the projected point
        """

        # reshape to get homogeneus transform
        if len(point) == 3:
            point = np.array([*point, 1])
        persp_m = np.array(self.projectionMatrix).reshape((4, 4)).T
        view_m = np.array(self.viewMatrix).reshape((4, 4)).T

        # Perspective proj matrix
        world_pix_tran = persp_m @ view_m @ point
        world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
        world_pix_tran[:3] = (world_pix_tran[:3] + 1) / 2
        x, y = world_pix_tran[0] * self.width, (1 - world_pix_tran[1]) * self.height
        x, y = np.floor(x).astype(int), np.floor(y).astype(int)
        return (x, y)

    def deproject(self, point, depth_img, homogeneous=False):
        """
        Deprojects a pixel point to 3D coordinates
        Args
            point: tuple (u, v); pixel coordinates of point to deproject
            depth_img: np.array; depth image used as reference to generate 3D coordinates
            homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                         else returns the world coordinates (x, y, z) position
        Output
            (x, y): np.array; world coordinates of the deprojected point
        """
        T_world_cam = np.linalg.inv(np.array(self.viewMatrix).reshape((4, 4)).T)

        u, v = point
        z = depth_img[v, u]
        foc = self.height / (2 * np.tan(np.deg2rad(self.fov) / 2))
        x = (u - self.width // 2) * z / foc
        y = -(v - self.height // 2) * z / foc
        z = -z
        world_pos = T_world_cam @ np.array([x, y, z, 1])
        if not homogeneous:
            world_pos = world_pos[:3]
        return world_pos
