#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg as LA
from matplotlib.patches import Ellipse
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal

K = 3
NUM_DATAPTS = 150

X, y = make_blobs(n_samples = NUM_DATAPTS, centers=K, shuffle=False, 
                   random_state =0, cluster_std=0.6)
g1 = np.asarray([[2.0, 0], [-0.9, 1.0]])
g2 = np.asarray([[1.4, 0], [0.5, 0.7]])
mean1 = np.mean(X[:int(NUM_DATAPTS/K)])
mean2 = np.mean(X[int(NUM_DATAPTS/K):2*int(NUM_DATAPTS/K)])
X[:int(NUM_DATAPTS/K)] = np.einsum('nj, ij -> ni', 
                                   X[:int(NUM_DATAPTS/K)] - mean1, g1) + mean1
X[int(NUM_DATAPTS/K):2*int(NUM_DATAPTS/K)] = np.einsum('nj,ij -> ni', 
                                                       X[int(NUM_DATAPTS/K):2*int(NUM_DATAPTS/K)] - mean2, g2) + mean2
X[:,1] -= 4


# # QUESTION 4 (a)

# In[53]:


mu = 5*np.random.random_sample((K, 2))-2.5
#for some godforsaken reason random.random.rand(K,2) does not work for me and im about to cry
cov = np.array([np.identity(2) for i in range(K)])
pi = np.array([1/K for i in range(K)])


# # QUESTION 4 (b)

# In[54]:


def E_step():
    gamma = np.zeros((NUM_DATAPTS,K))
    for i in range(NUM_DATAPTS):
        for j in range(K):
            gamma[i,j] = pi[j]*multivariate_normal.pdf(X[i],mu[j],cov[j])
        gamma[i,:] /= np.sum(gamma[i,:])
    return gamma


# # QUESTION 4 (c)

# In[55]:


def M_step(gamma):
    for i in range(K):
        counts = np.sum(gamma[:,i], axis=0)
        pi[i] = counts/NUM_DATAPTS
        wsum = np.sum(X*gamma[:,i].reshape(NUM_DATAPTS,1), axis=0)
        mu[i] = wsum/counts
        cov[i] = np.dot((np.array(gamma[:,i]).reshape(NUM_DATAPTS,1)*(X-mu[i])).T, (X-mu[i]))/counts
    #print(mu,"\n\n",cov,"\n\n",pi)
    return mu, pi, cov


# # QUESTION 4 (d)

# In[56]:


def plot_result(gamma=None):
    ax = plt.subplot(111, aspect='equal')
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])
    ax.scatter(X[:, 0], X[:, 1], c=gamma, s=50, cmap=None)
    
    for k in range(K):
        l, v = LA.eig(cov[k])
        theta = np.arctan(v[1,0]/v[0,0])
        e = Ellipse((mu[k, 0], mu[k, 1]), 6*l[0], 6*l[1], theta*180/np.pi)
        e.set_alpha(0.5)
        ax.add_artist(e)
        
    plt.show()


# In[57]:


# now write a loop that iterates through the E and M steps,
# and terminates after the change in log-likelihood is below some threshold
# at each iteration, print out the log-likelihood and use plot_result to plot progress of algo

def loop(threshold):
    maxiter = 100
    new_lh = 0
    for iteration in range(maxiter):
        old_lh = new_lh
        new_lh = 0
        estep = E_step()
        mu, pi, cov = M_step(estep)
        for i in range(NUM_DATAPTS):
            lhsum = 0
            for j in range(K):
                lhsum += pi[j]*multivariate_normal.pdf(X[i],mu[j],cov[j])
            new_lh += np.log(lhsum)
        print("Iteration results: \n  New Likelihood: ", new_lh, "\n  Change in Likelihood: ", new_lh-old_lh)
        plot_result(estep)
        if (new_lh - old_lh) < threshold and iteration>1:
            print("Ending conditions satisfied. Loop will end.")
            break


# In[58]:


loop(1)


# In[43]:


# people who have helped me suffer through this:
# 1) andrew tay 1002038, who was suffering with me while ploughing through several websites 
#      and slides next to me in capstone room, figuring out explanations and concepts together
# 2) cheong kaixiang 1002046, who very kindly and patiently helped me debug my code that was  
#      running into all sorts of weird errors to the best of his abilities
# 3) my instagram followers who consoled and encouraged me while I posted instagram 
#      stories of my failures

# online resources:
# https://zhiyzuo.github.io/EM/#algorithm-operationalization
# https://github.com/bdanalytics/UofWashington-MachineLearning/blob/master/clustering_3_em-for-gmm_BBI.py


# In[ ]:




