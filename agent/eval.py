import laser_env

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tqdm import trange


if __name__ == '__main__':
    env = make_vec_env('laser-v1', n_envs=1)
    model = PPO.load('model.pt')
    rewards = []
    distances = []
    angles = []
    init_distances = []
    init_angles = []
    for episode in trange(100):
        obs, tot_reward, done, info = env.reset(), 0, False, env.env_method('get_info')
        init_distances.append(info[0]['distance'])
        init_angles.append(info[0]['angle'])
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            tot_reward += reward
        rewards.append(tot_reward)
        distances.append(info[0]['distance'])
        angles.append(info[0]['angle'])
    print(f'reward: {np.mean(rewards):0.2f} +- {np.std(rewards):0.2f}')
    print(f'final distance: {np.mean(distances):0.2f} +- {np.std(distances):0.2f}')
    print(f'final angle: {np.mean(angles):0.2f} +- {np.std(angles):0.2f}')
    print(f'init distance: {np.mean(init_distances):0.2f} +- {np.std(init_distances):0.2f}')
    print(f'init angle: {np.mean(init_angles):0.2f} +- {np.std(init_angles):0.2f}')
