import laser_env

import numpy as np
from stable_baselines3 import PPO
import gym
from tqdm import trange


if __name__ == '__main__':
    env = gym.make('laser-v1')
    model = PPO.load('model.pt')
    rewards = []
    distances = []
    angles = []
    init_distances = []
    init_angles = []
    for episode in trange(100):
        obs, tot_reward, done, info = env.reset(), 0, False, env.get_info()
        init_distances.append(info['distance'])
        init_angles.append(info['angle'])
        while not done:
            action, _states = model.predict([obs], deterministic=True)
            obs, reward, done, info = env.step(action[0])
            tot_reward += reward
        rewards.append(tot_reward)
        distances.append(info['distance'])
        angles.append(info['angle'])
    print(f'reward: {np.mean(rewards):0.5f} +- {np.std(rewards):0.5f}')
    print(f'init distance: {np.mean(init_distances):0.5f} +- {np.std(init_distances):0.5f}')
    print(f'final distance: {np.mean(distances):0.5f} +- {np.std(distances):0.5f}')
    print(f'init angle: {np.mean(init_angles):0.5f} +- {np.std(init_angles):0.5f}')
    print(f'final angle: {np.mean(angles):0.5f} +- {np.std(angles):0.5f}')
