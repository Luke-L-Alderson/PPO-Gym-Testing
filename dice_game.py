import numpy as np
import scipy as sp
import random as rand
from matplotlib import pyplot as plt

# look into DQN for this, continuous issue

def egreedy(Q, epsilon):
    # given action values for a given state, returns the egreedy action.
    p = rand.random()
    if p <= epsilon:
        #print("Exploring")
        #print(np.random.randint(0, len(Q)))
        return np.random.randint(0, len(Q))
    else:
        #print("Exploiting")
        #print(np.argmax(Q))
        return np.argmax(Q)

num_episodes = 1000000
Q = np.random.rand(1000, 2)
episode_r = []
alpha = 0.99
gamma = 0.1
eps = 0.05

for episode in range(num_episodes):
    stop = False
    old_state = 0
    r = 0
    total = 0
    flag = 0
    #eps = 1/(episode+1)
    while True:
        action = egreedy(Q[old_state], eps)
        if action == 0:
            r = rand.randint(1,6) + rand.randint(1,6)
            new_state = old_state + r
            
            if new_state % 10 == 0 and new_state != 0:
                r = -50
                stop = True
                
        else:
            r = -5
            stop = True
            new_state = old_state
        
        total += r
        
        Q[old_state, action] = Q[old_state, action] + alpha*(r+gamma*np.max(Q[new_state])-Q[old_state, action])
        
        print(f"Stop ({stop}) - Old State ({old_state}) - Reward ({r}) - Action ({action}) - New State ({new_state})")
        old_state = new_state
        
        if stop is True:
            break
        
    episode_r.append(total)
    
plt.plot(episode_r)
plt.title("r")
plt.figure()  
plt.plot(Q[:,0])
plt.title("value function - action 0 rolling")
plt.figure() 
plt.plot(Q[:,1])
plt.title("value function - action 1 stopping")       
        
        
       
