# Derived from vapo_agent.py

import numpy as np

from vapo.affordance.utils.utils import get_transforms
from vapo.agent.core.target_search import TargetSearch
from vapo.agent.core.utils import tt


class VAPOAgent():
    def __init__(self, cfg):
        _aff_transforms = get_transforms(cfg.affordance.transforms.validation, cfg.target_search.aff_cfg.img_size)

        # To enumerate static cam preds on target search
        self.no_detected_target = 0

        args = {"initial_pos": self.origin, "aff_transforms": _aff_transforms, **cfg.target_search}
        _class_label = self.get_task_label()
        self.target_search = TargetSearch(self.env, class_label=_class_label, **args)

        # Target specifics
        self.env.target_search = self.target_search
        self.env.curr_detected_obj, _ = self.target_search.compute(rand_sample=True)
        self.eval_env = self.env
        self.radius = self.env.termination_radius  # Distance in meters
        self.sim = True

    def get_task_label(self):
        task = self.env.task
        if task == "hinge":
            return 1
        elif task == "drawer":
            return 2
        elif task == "slide":
            return 3
        else:  # pickup
            return None


    # Model based methods
    def detect_and_correct(self, env, obs, noisy=False, rand_sample=True):
        if obs is None:
            obs = env.reset()
        # Compute target in case it moved
        # Area center is the target position + 5cm in z direction
        env.move_to_target(self.origin)
        target_pos, no_target = self.target_search.compute(env, noisy=noisy, rand_sample=rand_sample)
        if no_target:
            self.no_detected_target += 1
        res = self.correct_position(env, obs, target_pos, no_target)
        return res

    def correct_position(self, env, s, target_pos, no_target):
        # Set current_target in each episode
        env.curr_detected_obj = target_pos
        env.move_to_target(target_pos)
        # as we moved robot, need to update target and obs
        # for rl policy
        return env, env.observation(env.get_obs()), no_target


    # Only applies to tabletop
    def tidy_up(self, env, max_episode_length=100):
        tasks = []
        # get from static cam affordance
        if env.task == "pickup":
            tasks = self.env.scene.table_objs
            n_tasks = len(tasks)

        ep_success = []
        total_ts = 0
        env.reset()
        # Set total timeout to timeout per task times all tasks + 1
        while (
            total_ts <= max_episode_length * n_tasks and self.no_detected_target < 3 and not self.env.all_objs_in_box()
        ):
            episode_length, episode_return = 0, 0
            done = False
            # Search affordances and correct position:
            env, s, no_target = self.detect_and_correct(env, self.env.get_obs(), rand_sample=True)
            if no_target:
                # If no target model will move to initial position.
                # Search affordance from this position again
                env, s, no_target = self.detect_and_correct(env, self.env.get_obs(), rand_sample=True)

            # If it did not find a target again, terminate everything
            while episode_length < max_episode_length and self.no_detected_target < 3 and not done:
                # sample action and scale it to action space
                s = env.transform_obs(tt(s), "validation")
                a, _ = self._pi.act(s, deterministic=True) # SAC action
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a) # environment step for task verification
                s = ns
                episode_return += r
                episode_length += 1
                total_ts += 1
                success = info["success"]
            ep_success.append(success)
            env.episode += 1
            env.obs_it = 0
        self.log.info("Success: %d/%d " % (np.sum(ep_success), len(ep_success)))
        return ep_success
