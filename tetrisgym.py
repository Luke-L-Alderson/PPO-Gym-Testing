import cv2
import gymnasium as gym
from helpers import save_gif
from tetris_gymnasium.wrappers.observation import RgbObservation
import matplotlib.pyplot as plt
import numpy as np
from tetris_gymnasium.mappings.rewards import RewardsMapping


if __name__ == "__main__":
    rewardmap = RewardsMapping(game_over=-10, alife=0, clear_line=1)
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human", rewards_mapping = rewardmap)
    
    env.reset(seed=42)
    
    env = RgbObservation(env)
    env.render_scaling_factor = 10
    observation_list = []
    terminated = False
    while not terminated:
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        key = cv2.waitKey(100) # timeout to see the movement
        observation_list.append(observation)
        print(reward)
        
    plt.imshow(observation)
    
    observation_list = np.stack(observation_list)
    save_gif(observation_list, filename="demo.gif", fps=10, resize_to=(600, 400))
    
    print("Game Over!")