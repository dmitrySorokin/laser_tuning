import laser_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


if __name__ == '__main__':
    env = make_vec_env('laser-v1', n_envs=16)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=500000, callback=lambda *args: model.save('model.pt'))
    model.save('model.pt')
