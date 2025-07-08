from network_definitions import PolicyNetwork
import torch
import torch.optim as optim
from env import DiceGame, run_episode, update_network
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import gymnasium as gym
from learners import reinforce_learner, a2c_learner, ppo_learner
import numpy as np
import optuna
from plotly.io import show
import gc
import os
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def objective(trial):
    
    params = {'episodes': 2000,
              'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
              'gamma': trial.suggest_float('gamma', 0.8, 0.99),
              'trunc': 200,
              'rollout_len': trial.suggest_int('rollout_len', 8, 512, step = 8, log=True),
              'eps': trial.suggest_float('eps', 0.1, 0.3),
              'update_epochs': 1,
              'iterations': 5000}
    
    #print(f"lr: {params['lr']}\ngamma: {params['gamma']}")
    print(params)
    env = gym.make('CartPole-v1', max_episode_steps = params.get("trunc"))
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
    
    
    if one_off:
       params_of = {'episodes': 2000,
                 'lr': 6e-4,
                 'gamma': 0.85,
                 'trunc': 200,
                 'rollout_len': 2**6,
                 'eps': 0.2,
                 'update_epochs': 1,
                 'iterations': 5000}
       
       print(f"lr: {params_of['lr']}\ngamma: {params_of['gamma']}")
       
       # Make environment, run actor-critic learner, plot rewards over episodes
       env = gym.make('CartPole-v1', max_episode_steps = params_of.get("trunc"))
       a2c_reward = ppo_learner(env, params_of, device)
       plt.plot(a2c_reward)
       plt.show()
       
    else:
       # Run a hyperparameter sweep using params in the objective function
       study = optuna.create_study(direction='maximize')
       study.optimize(objective, n_trials=15, gc_after_trial=True)
       fig = optuna.visualization.plot_intermediate_values(study)
       show(fig)
       plt.savefig(fig)
