#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

T = 1000


# # QUESTION 3 (a)

# In[2]:


# Q: array representing the action-values, N: an array representing the number of times different
# actions have been taken, t: total num actions taken thus far, policy-specific parameter

def e_greedy(Q, N, t, e):
    if np.random.random() < e:
        actions = np.delete(np.copy(Q), np.argmax(Q))
        return np.random.choice(len(actions))
    else:
        return np.argmax(Q)
    
def UCB (Q, N, t, c):
    actions = []
    for i in range(len(Q)):
        actions.append(Q[i] + c*np.sqrt(np.log(t+1)/(N[i]+1)))
    return np.argmax(actions)


# # QUESTION 3 (b)

# In[3]:


# one run = 1000 time steps; updates Q incrementally & records the reward received at each time step
# at each time step: when action a taken, reward r is sampled.

def test_run(policy, param):
    true_means = np.random.normal(0,1,10)
    reward = np.zeros(T+1)
    Q = np.zeros(10)
    N = np.zeros(10)
    for t in range(T):
        a = policy(Q,N,t,param)
        r = np.random.normal(true_means[a],1)
        reward[t+1] = r
        Q[a] = Q[a] + 1/(N[a]+1)*(r-Q[a])
        N[a] += 1
    return reward


# # QUESTION 3 (c)

# In[18]:


def main ():
    ave_g = np.zeros(T+1)
    ave_eg = np.zeros(T+1)
    ave_ucb = np.zeros(T+1) 

    for i in range(2000):
        g = test_run(e_greedy , 0.0)
        eg = test_run(e_greedy , 0.07)
        ucb = test_run(UCB, 1.73)
        ave_g += (g - ave_g ) / (i + 1)
        ave_eg += (eg - ave_eg) / (i + 1)
        ave_ucb += (ucb - ave_ucb) / (i + 1)
    
    t = np.arange(T + 1)
    plt.plot(t, ave_g, 'b-',t , ave_eg , 'r-' , t , ave_ucb , 'g-')
    plt.show()


# In[19]:


main()


# #### At approximately eps =   the eps-greedy method switches from being better than greedy policy to being worse than it.

# # Help & References

# In[ ]:


# Andrew Tay 1002038 for checking+debugging through my code on top of brainstorming with me in the 
#    capstone room over the weekend (& mon & tues & wed) once again in shared attempts to do HW5
# Benedict See 1002014 for discussing with us on a fine desperate wednesday night

# https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Control%20with%20Epsilon-Greedy%20Policies%20Solution.ipynb

