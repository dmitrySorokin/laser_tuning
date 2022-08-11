import gym
import numpy as np
import time as tm
from .utils import reflect, project, rotate_x, dist, angle_between


class LaserEnv(gym.Env):
    # mirror screw step l / L, (ratio of delta screw length to vertical distance)
    one_mirror_step = 0.52 * 1e-6
    mirror_max_screw_value = one_mirror_step * 5000

    metadata = {'render.modes': ['human', 'rgb_array']}

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    # initial normals
    mirror1_x_rotation_angle = 3 * np.pi / 4
    mirror2_x_rotation_angle = -3 * np.pi / 4

    def __init__(self, a=38.5, b=40, c=38.5):
        """
        camera <---------\
                         ^
                         |
                         |
        laser ---------> /
        dist(laser, mirror1) = a
        dist(mirror1, mirror2) = b
        dist(mirror2, camera) = c

        axis:

        ^ y
        |
        |
        |
        ----------> z
        """
        # size of interferometer (in mm)
        self.a = a
        self.b = b
        self.c = c

        self.mirror1_screw_x = 0
        self.mirror1_screw_y = 0
        self.mirror2_screw_x = 0
        self.mirror2_screw_y = 0

        self.state = None
        self.n_steps = None
        self.info = None
        self.max_steps = 100

    def get_info(self):
        return self.info

    def seed(self, seed=None):
        self.action_space.seed(seed)

    def step(self, actions):
        """
        :param action: (delta_11, delta_12, delta_21, delta_22)
        :return: (state, reward, done, info)
        """

        self.n_steps += 1
        self.info = {}

        for action_id, action_value in enumerate(actions):
            self._take_action(action_id, action_value)

        center, wave_vector = self._calc_center_and_wave_vector()
        proj, dist = self._calc_projection_distance(center, wave_vector)
        self.info['proj'] = proj
        self.info['distance'] = dist

        self.state = self._calc_state(center, wave_vector, proj)
        reward = self._calc_reward(dist)

        return self.state, reward, self.game_over(), self.info

    def reset(self, actions=None):
        self.n_steps = 0
        self.info = {}

        self.mirror1_screw_x = 0
        self.mirror1_screw_y = 0
        self.mirror2_screw_x = 0
        self.mirror2_screw_y = 0

        if actions is None:
            actions = LaserEnv.action_space.sample()

        for action_id, action_value in enumerate(actions):
            self._take_action(action_id, action_value)

        center, wave_vector = self._calc_center_and_wave_vector()
        proj, dist = self._calc_projection_distance(center, wave_vector)
        self.info['proj'] = proj
        self.info['distance'] = dist

        self.state = self._calc_state(center, wave_vector, proj)
        return self.state

    def render(self, mode='human', close=False):
        assert False, 'not implemented'

    def _take_action(self, action, normalized_step_length):
        if action == 0:
            self.mirror1_screw_x = np.clip(self.mirror1_screw_x + normalized_step_length, -1, 1)
        elif action == 1:
            self.mirror1_screw_y = np.clip(self.mirror1_screw_y + normalized_step_length, -1, 1)
        elif action == 2:
            self.mirror2_screw_x = np.clip(self.mirror2_screw_x + normalized_step_length, -1, 1)
        elif action == 3:
            self.mirror2_screw_y = np.clip(self.mirror2_screw_y + normalized_step_length, -1, 1)
        else:
            assert False, 'unknown action = {}'.format(action)

    def _calc_center_and_wave_vector(self):
        assert np.abs(self.mirror1_screw_x) <= 1, self.mirror1_screw_x
        assert np.abs(self.mirror1_screw_y) <= 1, self.mirror1_screw_y
        assert np.abs(self.mirror2_screw_x) <= 1, self.mirror2_screw_x
        assert np.abs(self.mirror2_screw_y) <= 1, self.mirror2_screw_y

        mirror1_screw_x_value = self.mirror1_screw_x * LaserEnv.mirror_max_screw_value
        mirror1_screw_y_value = self.mirror1_screw_y * LaserEnv.mirror_max_screw_value
        mirror1_x_component = - mirror1_screw_x_value / np.sqrt(mirror1_screw_x_value ** 2 + 1)
        mirror1_y_component = - mirror1_screw_y_value / np.sqrt(mirror1_screw_y_value ** 2 + 1)
        mirror1_z_component = np.sqrt(1 - mirror1_x_component ** 2 - mirror1_y_component ** 2)
        mirror1_normal = np.array(
            [mirror1_x_component, mirror1_y_component, mirror1_z_component],
            dtype=np.float64
        )
        mirror1_normal = rotate_x(mirror1_normal, LaserEnv.mirror1_x_rotation_angle)

        mirror2_screw_x_value = self.mirror2_screw_x * LaserEnv.mirror_max_screw_value
        mirror2_screw_y_value = self.mirror2_screw_y * LaserEnv.mirror_max_screw_value
        mirror2_x_component = - mirror2_screw_x_value / np.sqrt(mirror2_screw_x_value ** 2 + 1)
        mirror2_y_component = - mirror2_screw_y_value / np.sqrt(mirror2_screw_y_value ** 2 + 1)
        mirror2_z_component = np.sqrt(1 - mirror2_x_component ** 2 - mirror2_y_component ** 2)
        mirror2_normal = np.array(
            [mirror2_x_component, mirror2_y_component, mirror2_z_component],
            dtype=np.float64
        )
        mirror2_normal = rotate_x(mirror2_normal, LaserEnv.mirror2_x_rotation_angle)

        self.info['mirror1_normal'] = mirror1_normal
        self.info['mirror2_normal'] = mirror2_normal

        center = np.array([0, -self.b, self.c - self.a], dtype=np.float64)
        wave_vector = np.array([0, 0, 1], dtype=np.float64)

        # reflect wave vector by first mirror
        center = project(center, wave_vector, mirror1_normal, np.array([0, -self.b, self.c]))
        wave_vector = reflect(wave_vector, mirror1_normal)
        self.info['reflect_with_mirror1'] = 'center = {}, k = {}'.format(center, wave_vector)

        # reflect wave vector by second mirror
        center = project(center, wave_vector, mirror2_normal, np.array([0, 0, self.c]))
        wave_vector = reflect(wave_vector, mirror2_normal)
        self.info['reflect_with_mirror2'] = 'center = {}, k = {}'.format(center, wave_vector)

        self.info['kvector'] = wave_vector

        self.info['angle'] = angle_between(wave_vector, [0, 0, -1])

        return center, wave_vector

    def _calc_projection_distance(self, center, wave_vector):
        projection_plane_normal = np.array([0, 0, 1])
        projection_plane_center = np.array([0, 0, 0])

        proj = project(center, wave_vector, projection_plane_normal, projection_plane_center)
        distance = dist(proj, [0, 0, 0])

        return proj, distance


    def game_over(self):
        return self.n_steps >= self.max_steps

    def _calc_state(self, center1, wave_vector1, proj_1):
        return proj_1[:2]

    def _calc_reward(self, dist):
        return -dist
