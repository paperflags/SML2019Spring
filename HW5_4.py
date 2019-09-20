#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
from gym.envs.toy_text import discrete

UP=0
RIGHT=1
DOWN=2
LEFT=3

GOAL=4     # upper-right corner
START=20   # lower-left corner
SNAKE1=7
SNAKE2=17

eps=0.25

class Robot_vs_snakes_world(discrete.DiscreteEnv):
    def __init__(self):
        self.shape=[5, 5]
        nS = np.prod(self.shape)  # total num of states
        nA = 4                    # total num of actions per state
        MAX_Y = self.shape[0]
        MAX_X = self.shape[1]
        
        P={}
        grid=np.arange(nS).reshape(self.shape)
        it=np.nditer(grid, flags=['multi_index'])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            P[s] = {a: [] for a in range(nA)}
            is_done = lambda s: s == GOAL
            
            if is_done(s):
                reward = 0.0
            elif s == SNAKE1 or s == SNAKE2:
                reward = -15.0
            else :
                reward = -1.0
            
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s+1
                ns_down = s if y == (MAX_Y - 1) else s+MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1 - (2*eps), ns_up, reward, is_done(ns_up )),
                            (eps, ns_right, reward, is_done(ns_right)),
                            (eps, ns_left, reward, is_done(ns_left))]
                P[s][RIGHT] = [(1 - (2 * eps), ns_right, reward, is_done(ns_right)),
                               (eps, ns_up, reward, is_done(ns_up)),
                               (eps, ns_down, reward, is_done(ns_down))]
                P[s][DOWN] = [(1 - (2 * eps), ns_down, reward, is_done(ns_down)),
                              (eps, ns_right, reward, is_done(ns_right)),
                              (eps, ns_left, reward, is_done(ns_left))]
                P[s][LEFT] = [(1 - (2 * eps), ns_left, reward, is_done(ns_left)),
                              (eps, ns_up, reward, is_done(ns_up)),
                              (eps, ns_down, reward, is_done(ns_down))]
            it.iternext()
                
        isd = np.zeros(nS)
        isd[START] = 1.0
        
        self.P = P

        super(Robot_vs_snakes_world, self).__init__(nS, nA, P, isd)

    def _render(self):
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " R "
            elif s == GOAL:
                output = " G "
            elif s == SNAKE1 or s == SNAKE2:
                output = " S "
            else:
                output = " o "
            
            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()
            
            sys.stdout.write(output)
            
            if x == self.shape[1]-1:
                sys.stdout.write("\n")
                
            it.iternext()
            
        sys.stdout.write("\n")


# In[2]:


env = Robot_vs_snakes_world()


# # QUESTION 4 (a)

# In[3]:


def value_iteration(env):
    policy = np.zeros([env.nS, env.nA])
    V = np.zeros(env.nS)
    max_iter = 1000
    threshold = 1e-4
    
    for i in range(max_iter):
        V_old = np.copy(V)
        for s in range(env.nS):
            q = [0]*env.nA
            for a in range(env.nA):
                for j in env.P[s][a]:
                    q[a]+=(j[0]*(j[2] + V_old[j[1]]))           
            V[s] = max(q)
        
        if (all(i<threshold for i in np.fabs(V - V_old)) == True):
            break
    
    for s in range(env.nS):
        q = [0]*env.nA
        for a in range(env.nA):
            for j in env.P[s][a]:
                q[a]+=(j[0]*(j[2] + V[j[1]]))
        policy[s] = np.argmax(q)
    
    # print out both the policy and value function
    return policy, V

policy, V = value_iteration(env)
print("Policy Function is: \n", policy, "\n Value Function is: \n", V)


# # QUESTION 4 (b)

# In[4]:


# use policy in (a) to nav the robot; at each step, use env._render() to print
# terminate program when robot reaches exit

total_rew = 0
step = 0
snake_enc = 0
policy, V = value_iteration(env)
directions = ["UP", "RIGHT", "DOWN", "LEFT"]

while True:
    print("-------------------------  Step number: ", step)
    env._render()
    s = env.s
    print("Step instructed: ", directions[int(policy[s][0])])
    s, reward, done, _ = env.step((policy[s][0]))
    total_rew += reward
    if s == 7 or s == 17:
        snake_enc += 1
    step += 1
    if done:
        print("\n Total Steps: ", step, "\n Snake Encounters: ", snake_enc, "\n Hit Points Lost: ", step+15*snake_enc, "\n Total reward: ", total_rew)
        break


# # QUESTION 4 (c)

# In[5]:


# does the robot try to avoid the snakes? if so, refer to the policy function obtained
# in (a) to explain in what way it does so


# In[9]:


for i in range(100):
    env = Robot_vs_snakes_world()

    total_rew = 0
    step = 0
    snake_enc = 0
    policy, V = value_iteration(env)
    directions = ["UP", "RIGHT", "DOWN", "LEFT"]

    while True:
        s = env.s
        s, reward, done, _ = env.step((policy[s][0]))
        total_rew += reward
        if s == 7 or s == 17:
            snake_enc += 1
        step += 1
        if done:
            if snake_enc > 0:
                print("Run", i, ": snakes =", snake_enc)
            break


# #### Since most of the runs are without any snakes, it seems like the robot does try to avoid the snakes. In part (a), the policy for a state is selected by finding the direction that gives the maximum reward after calculating the reward for each direction. Moving into a room with a snake will always result in a greater penalty than moving into a room without, the policy will always choose to move in a direction that avoids snakes.

# In[ ]:


# Andrew Tay 1002038 for checking+debugging through my code on top of brainstorming with me in the 
#    capstone room over the weekend (& mon & tues & wed) once again in shared attempts to do HW5
# Benedict See 1002014 for discussing with us on a fine desperate wednesday night

