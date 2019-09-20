#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct


# In[2]:


def func(x1,x2):
    return 0.5*np.exp(-0.5*((x1+1.25)**2+(x2+1.75)**2))+np.exp(-0.5*((x1-2.25)**2+(x2-2.65)**2))

def noisy_func(x1,x2):
    output=func(x1,x2)
    noise=np.random.normal(0,0.1,np.shape(output))
    return output+noise


# # Q5 (a)

# In[3]:


#acquisition functions: probability of improvement
def probability_of_improv(mux, stdx, fmax):
    Z = (mux - fmax)/stdx
    return norm.cdf(Z)

#expected improvement
def expected_improv(mux, stdx, fmax):
    Z = (mux - fmax)/stdx
    E = (mux - fmax)*norm.cdf(Z) + stdx*norm.pdf(Z)
    E[stdx == 0.0] = 0.0
    return E

#upper confidence bound
def upp_conf_bound(mux, stdx, t, d, v=1, delta=.1):
    K = np.sqrt(v*(2*np.log((t**(d/2.0+2))*(np.pi**2)/(3.0*delta))))
    UCB = mux+K*stdx
    return UCB


# # Q5 (b)

# In[4]:


def query(opt_val,gp):
    def obj(x):
        mu,std=gp.predict(x.reshape(-1,2),return_std=True)
        return -expected_improv(mu, std, opt_val)
    res=minimize(obj, np.random.random(2))
    return res.x


# In[5]:


res=50
lin=np.linspace(-5,5,res)
meshX,meshY=np.meshgrid(lin,lin)
meshpts=np.vstack((meshX.flatten(),meshY.flatten())).T
def add_subplot(gp,subplt):
    mu = gp.predict(meshpts,return_std=False)
    ax = fig.add_subplot(2,5,subplt,projection='3d')
    ax.plot_surface(meshX,meshY,np.reshape(mu,(50,50)),
                    rstride=1,cstride=1,cmap=cm.jet,linewidth=0, antialiased=False)
    
if __name__=='__main__':
    true_y = func(meshX,meshY)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(meshX,meshY,true_y,rstride=1,cstride=1,cmap=cm.jet,linewidth=0,antialiased=False)
    plt.title('True_function')
    plt.show()
    
    fig=plt.figure(figsize=plt.figaspect(0.5))


# # Q5 (c)

# In[6]:


#initialize 4 random points and evaluate the noisy function at these points

xi = np.random.random(size=(4,2))
yi = np.array([noisy_func(x[0],x[1]) for x in xi])


# # Q5 (d)

# In[7]:


gp = GaussianProcessRegressor()
fig = plt.figure()
for i in range(10):
    gp.fit(xi,yi)
    #find the current optimal value and its location
    opt_val = np.amax(yi)
    opt_x = xi[np.argmax(yi)]
    print('Best Value: ', opt_val)
    print('at ', opt_x)
    
    next_x = query(opt_val,gp)
    
    #add next_x to the list of data points
    xi=np.append(xi,next_x).reshape(-1,2)
    
    next_y=noisy_func(xi[-1][0],xi[-1][1]).reshape(1)
    
    #add next_y to the list of observations
    yi = np.append(yi, next_y)
    add_subplot(gp, i+1)
    
plt.show()


# In[8]:


#I am far from optimal value and there was only slight improvement from the first iteration.


# In[ ]:




