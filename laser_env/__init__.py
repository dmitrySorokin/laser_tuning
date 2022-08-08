from gym.envs.registration import register
from .envs import LaserEnv

register(id='laser-v1',
         entry_point='laser_env.envs:LaserEnv')
