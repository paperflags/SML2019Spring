#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs, make_circles


# In[2]:


def plot_svc_decision(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y, X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X,Y,P,colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])
    # plot supprt vectors
    ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],
              s=300,linewidth=1,edgecolors='black',facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# # Q4 (a)

# In[5]:


X, y = make_circles(100,factor=.1,noise=.1)
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.scatter(X[:,0],X[:,1],c=y,s=50,cmap='seismic')

model = SVC(kernel='rbf',C=1E10)
model.fit(X,y)
plot_svc_decision(model, ax1)

#use SVC to construct a support vector machine (will have to specify a
#kernel and the regularization parameter C) to classify this dataset, then
#use fit(X,y) to feed in the data and labels. Show your results using the
#plot_svc_decision function


# # Q4 (b)

# In[4]:


X, y = make_blobs(n_samples=100,centers=2,random_state=0,cluster_std=1.0)
fig2 = plt.figure(figsize=(16,6))
fig2.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
ax2=fig2.add_subplot(121)
ax2.scatter(X[:,0],X[:,1],c=y,s=50,cmap='seismic')
ax3=fig2.add_subplot(122)
ax3.scatter(X[:,0],X[:,1],c=y,s=50,cmap='seismic')

model1 = SVC(kernel='rbf',C=1E5)
model1.fit(X,y)
plot_svc_decision(model1,ax2)

model2 = SVC(kernel='rbf',C=1E2)
model2.fit(X,y)
plot_svc_decision(model2,ax3)

#Your task here is to classify the dataset using different values of the
#regularization parameter C to understand soft margins in SVM. Indicate clearly
#what values of C you are using, and plot your results with plot_svc_decision
#using ax2 for one model and ax3 for the other


# In[ ]:




