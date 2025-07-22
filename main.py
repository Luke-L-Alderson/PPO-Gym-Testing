import torch
from matplotlib import pyplot as plt
import optuna
from plotly.io import show, renderers
import os
from learners import PPOagent
from helpers import convert_to_one_dict, moving_avg
import statistics as stats
import json
import numpy as np

renderers.default='browser'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def objective(trial):
    
    params = {'training_steps': 100000,
              'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
              'gamma': trial.suggest_float('gamma', 0.8, 0.99),
              'trunc': 400,
              'rollout_len': trial.suggest_int('rollout_len', 64, 1024, log=True),
              'eps': 0.2,
              'update_epochs': trial.suggest_int('update_epochs', 1, 10, log=False),
              'entropy': trial.suggest_float('entropy', 0, 0.1),
              'num_envs': trial.suggest_int('num_envs', 2, 10, step = 2, log=False),
              'annealing': trial.suggest_categorical("annealing", [True, False])}
    
    for (key, val) in params.items():
        print(f"{key} = {val}")
    
    agent = PPOagent(trial = trial,
                     training_steps=params.get("training_steps"),
                     lr = params.get("lr"),
                     gamma = params.get("gamma"),
                     trunc = params.get("trunc"),
                     rollout_len=params.get("rollout_len"),
                     update_epochs=params.get("update_epochs"),
                     entropy_coef=params.get("entropy"),
                     eps=params.get("eps"),
                     num_envs=params.get("num_envs"),
                     annealing=params.get("annealing"))
    
    ppo_reward = agent.train_agent()
    assert ppo_reward, "No rewards returned, try increasing training steps, or reducing num_envs * rollout_len"
    
    ppo_reward = convert_to_one_dict(ppo_reward)
   
    means = []
    for (key, val) in ppo_reward.items():
            means.append(stats.mean(val))

    return stats.mean(means)

if __name__ == '__main__':
    
    # Let PyTorch use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set this to False to enable a full hyperparameter sweep with optuna
    one_off = True
    
    # Set this True to use Stable Baselines PPO hyperparameters
    sb = False
    imp = True
    
    assert not (sb and imp), "Cannot use Stable Baselines and tuned-hyperparameters. Set one to False."
    
    if one_off:
       params = {'training_steps': 1e6,
                 'lr': 3e-4,
                 'gamma': 0.99,
                 'trunc': 400,
                 'rollout_len': 2048,
                 'eps': 0.2,
                 'update_epochs': 10,
                 'ent_coef': 0.00,
                 'num_envs': 4,
                 'lam': 0.95,
                 'val_coef': 0.5,
                 'glob_norm': 0.5,
                 'annealing': True}
       
       stable_baselines_params = {'training_steps': 1e6,
                                  'lr': 3e-4,
                                  'gamma': 0.99,
                                  'trunc': 400,
                                  'rollout_len': 2048,
                                  'eps': 0.2,
                                  'update_epochs': 10,
                                  'entropy': 0.0,
                                  'num_envs': 4,
                                  'lam': 0.95,
                                  'val_coef': 0.5,
                                  'ent_coef': 0.0,
                                  'glob_norm': 0.5}
       
       if sb:
           # use same parameters as Stable Baselines v3
           params.update(stable_baselines_params)
           sb_minibatch_size = 64
           print(f"Using Stable Baselines values. The number of batches is {params.get("rollout_len")*params.get("num_envs")//sb_minibatch_size} with a batch size of {sb_minibatch_size}")
       elif imp:
           # import good parameters identified during Optuna trials and merge with base params
           import_params = json.load(open("good_params.txt"))
           params.update(import_params)
       
       print(f"lr: {params['lr']}\ngamma: {params['gamma']}")
           
       agent = PPOagent(env_name="Acrobot-v1",
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
                        num_minibatches=params.get("rollout_len")*params.get("num_envs")//sb_minibatch_size if sb else 4,
                        value_coef=params.get("val_coef"),
                        global_gradient_norm=params.get("glob_norm"),
                        annealing=params.get("annealing"))
       
       ppo_reward = agent.train_agent()
       ppo_reward = convert_to_one_dict(ppo_reward)
       
       plotting_data = moving_avg(np.array(ppo_reward[0]), len(ppo_reward[0])//100)
       plt.plot(plotting_data)
       plt.show()
       
    else:
       # Run a hyperparameter sweep using params in the objective function
       study = optuna.create_study(direction='maximize')
       study.optimize(objective, n_trials=30, gc_after_trial=True)
       fig = optuna.visualization.plot_intermediate_values(study)
       show(fig)
       
       trial_dict = {}
       for i in range(len(study.trials)):
           themean = stats.mean(study.trials[i].intermediate_values.values())
           trial_dict[i] = themean
    
       json.dump(study.trials[max(trial_dict, key=trial_dict.get)].params, open("good_params.txt", "w"))   
       print(f"\nThe best trial was {max(trial_dict, key=trial_dict.get)} with parameters:\n{study.trials[max(trial_dict, key=trial_dict.get)].params}")