from network_definitions import PolicyNetwork
import torch
import torch.optim as optim
from env import DiceGame, run_episode, update_network
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation
from learners import reinforce_learner, a2c_learner, ppo_learner, ppo_learner_image, make_env
import numpy as np
import optuna
from plotly.io import show, renderers
import os
from gymnasium.wrappers import RecordVideo
from datetime import datetime
from helpers import save_gif

import plotly.express as px
from tetris_gymnasium.mappings.rewards import RewardsMapping
renderers.default='browser'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def objective(trial):
    
    params = {'episodes': 2000,
              'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
              'gamma': trial.suggest_float('gamma', 0.8, 0.99),
              'trunc': 400,
              'rollout_len': trial.suggest_int('rollout_len', 64, 128, step = 8, log=False),
              'eps': trial.suggest_float('eps', 0.1, 0.3),
              'update_epochs': 1,
              'iterations': 1000,
              'entropy': trial.suggest_categorical('entropy', [0, 0.001, 0.01, 0.02, 0.03, 0.1])}
    
    #print(f"lr: {params['lr']}\ngamma: {params['gamma']}")
    print(params)
    
    #rewardmap = RewardsMapping(game_over=-10, alife=0, clear_line=1)
    #env = gym.make("tetris_gymnasium/Tetris", render_mode="human", rewards_mapping = rewardmap)
    #env = RgbObservation(env)
    #env.render_scaling_factor = 10
    #a2c_reward = ppo_learner_image(env, params, device, trial)
    
    env = gym.make('CartPole-v1', max_episode_steps = params.get("trunc"))
    env = gym.vector.SyncVectorEnv([make_env()]*10)
    a2c_reward = ppo_learner(env, params, device, trial)
    
    return sum(a2c_reward)/len(a2c_reward)

if __name__ == '__main__':
    
    # Let PyTorch use GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
        
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    # Set this to False to enable a full hyperparameter sweep with optun
    one_off = True
    tetris = False
    
    if one_off:
       params_of = {'episodes': 2000,
                 'lr': 6e-4,
                 'gamma': 0.85,
                 'trunc': 200,
                 'rollout_len': 128,
                 'eps': 0.2,
                 'update_epochs': 1,
                 'iterations': 1000,
                 'entropy': 0.03}
       
       print(f"lr: {params_of['lr']}\ngamma: {params_of['gamma']}")
       
       # Make environment, run actor-critic learner, plot rewards over episodes
       if tetris:
           rewardmap = RewardsMapping(game_over=-10, alife=0, clear_line=1)
           env = gym.make("tetris_gymnasium/Tetris", render_mode="human", rewards_mapping = rewardmap)
           env = RgbObservation(env)
           env.render_scaling_factor = 10
           a2c_reward = ppo_learner_image(env, params_of, device)
       else:
           trigger = lambda t: t % 100 == 0
           timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
           env = gym.make('CartPole-v1', max_episode_steps = params_of.get("trunc"), render_mode='rgb_array')
           
           env = gym.vector.SyncVectorEnv([make_env()]*10)
           #env = RecordVideo(env, video_folder=f"./training_videos", episode_trigger=trigger, disable_logger=True)
           ppo_reward,  full_pixels = ppo_learner(env, params_of, device)
           #save_gif(full_pixels, filename="cartpole.gif")
       
       
       plt.plot(ppo_reward)
       plt.show()
       
    else:
       # Run a hyperparameter sweep using params in the objective function
       study = optuna.create_study(direction='maximize')
       study.optimize(objective, n_trials=15, gc_after_trial=True)
       fig = optuna.visualization.plot_intermediate_values(study)
       show(fig)
       plt.savefig(fig)
