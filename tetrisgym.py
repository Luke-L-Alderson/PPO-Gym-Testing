import cv2
import gymnasium as gym
from helpers import save_gif, convert_to_one_dict, moving_avg, make_env_t
from tetris_gymnasium.wrappers.observation import RgbObservation
from tetris_gymnasium.mappings.rewards import RewardsMapping
from gymnasium.wrappers import TransformObservation, RecordVideo, FrameStackObservation, GrayscaleObservation
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import to_grayscale
import torch
import os
from network_definitions import ConvAgent
from collections import defaultdict
from learners import PPOagent
from helpers import stack_frames
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def run_test_episode():
    t1 = v2.Compose([   #[envs, stack, H, W, C]
                        v2.Lambda(lambda a: torch.movedim(a, -1, -3)),              # [envs, stack, C, H, W]
                        v2.Lambda(lambda x: v2.functional.rgb_to_grayscale(x)),     # grayscale: [envs, stack, 1, H, W]
                        v2.Lambda(lambda x: x.squeeze(2) if x.shape[-3]==1 else stack_frames(x)),                          # remove channel dim: [envs, stack, H, W]
                        v2.ToDtype(torch.float32),
                        v2.Resize(84, 84)
                        ]) #[envs, C*stack, H, W]
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    env = gym.vector.SyncVectorEnv([make_env_t(environment="tetris_gymnasium/Tetris", frames=4)]*2)
    env = gym.wrappers.vector.RecordEpisodeStatistics(env)
    env = gym.wrappers.vector.NumpyToTorch(env, device)
    env = gym.wrappers.vector.TransformObservation(env, lambda x: t1(x))
    
    print("Running a random episode.")
    obs, info = env.reset(seed=42)
    valdict = defaultdict(list)
    terminated = torch.tensor([False, False], dtype=bool)
    truncated = torch.tensor([False, False], dtype=bool)
    while not any(terminated|truncated):
        action = torch.tensor(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        if not any(terminated|truncated):
            reward[~(terminated|truncated)]+=0.1
            
        valdict["obs"].append(obs)
        valdict["r"].append(reward)
        print(reward)
    
    print("Done!")
    print(terminated, truncated)
    return valdict, obs
    
if __name__ == "__main__":

    
    dummy = False
    
    if dummy:
        
        
        run_test_episode()
    else:
        params = {'lr': 3e-4,
                  'gamma': 0.99,
                  'trunc': 1e7,
                  'num_batches': 4,
                  'rollout_len': 2048,
                  'eps': 0.2,
                  'update_epochs': 8,
                  'ent_coef': 0.03,
                  'num_envs': 16,
                  'lam': 0.95,
                  'val_coef': 0.5,
                  'glob_norm': 0.5,
                  'annealing': False}
        
        params["training_steps"] = 1e5 * params.get("num_envs")
        
        print("Starting PPO on Tetris")
        agent = PPOagent(env_name="tetris_gymnasium/Tetris",
                         state_space_not_pixels=False,
                         training_steps=params.get("training_steps"),
                         lr = params.get("lr"),
                         gamma = params.get("gamma"),
                         trunc = params.get("trunc"),
                         rollout_len=params.get("rollout_len"),
                         update_epochs=params.get("update_epochs"),
                         entropy_coef=params.get("ent_coef"),
                         eps=params.get("eps"),
                         num_envs=params.get("num_envs"),
                         lam=params.get("lam"),
                         num_minibatches=params.get("num_batches"),
                         value_coef=params.get("val_coef"),
                         global_gradient_norm=params.get("glob_norm"),
                         annealing=params.get("annealing"))
        
        ppo_reward = agent.train_agent()
        ppo_reward = convert_to_one_dict(ppo_reward)
        
        plotting_data = moving_avg(np.array(ppo_reward[0]), len(ppo_reward[0])//100)
        plt.plot(plotting_data)
        plt.show()
        
        #plt.imshow(torch.movedim(final_obs[0], 0, -1).cpu())
        
        #obs_list = np.stack(obs_list)
        #save_gif(obs_list, filename="demo.gif", fps=10, resize_to=(600, 400))