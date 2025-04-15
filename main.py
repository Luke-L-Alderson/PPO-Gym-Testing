from network_definitions import PolicyNetwork
import torch
import torch.optim as optim
from env import DiceGame, run_episode, update_network
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import gym
from learners import reinforce_learner, a2c_learner
import numpy as np
import optuna

def objective(trial):
    
    params = {'episodes': 2000,
              'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
              'gamma': trial.suggest_uniform('gamma', 0.8, 0.99),
              'trunc': 200}
    
    print(f"lr: {params['lr']}\ngamma: {params['gamma']}")
    
    env = gym.make('CartPole-v1', max_episode_steps = params.get("trunc"))
    a2c_reward = a2c_learner(env, params, device)
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
    one_off = False
    
    if one_off:
       params = {'episodes': 2000,
                 'lr': 1e-3,
                 'gamma': 0.99,
                 'trunc': 200}
       
       print(f"lr: {params['lr']}\ngamma: {params['gamma']}")
       
       # Make environment, run actor-critic learner, plot rewards over episodes
       env = gym.make('CartPole-v1', max_episode_steps = params.get("trunc"))
       a2c_reward = a2c_learner(env, params, device)
       plt.plot(a2c_reward)
       plt.show()
       
    else:
       # Run a hyperparameter sweep using params in the objective function
       study = optuna.create_study(direction='maximize')
       study.optimize(objective, n_trials=10, gc_after_trial=True)