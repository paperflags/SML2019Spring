#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

labels = {
    0:"T-shirt",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle_boot"    
}


# In[2]:


# Q4 a)


# In[9]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #initiate layers here
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(7*7*32, 10)


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# In[3]:


# Q4 b)


# In[15]:


def main():
    NUMEPOCHS=20
    LRATE=0.001
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset=datasets.FashionMNIST('../data',train=True,
                                      download=True,transform=transforms.ToTensor())
    test_dataset=datasets.FashionMNIST('../data',train=False,
                                      download=True,transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True);
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False);
    model=CNN().to(device)
    optimizer=optim.SGD(model.parameters(),lr=LRATE)
    
    for epoch in range(1,NUMEPOCHS+1):
        test(model,device,test_loader)
        train(model,device,train_loader,optimizer,epoch)
        
    test(model,device,test_loader)
    
if __name__=='__main__':
    main()


# In[4]:


# Q4 c)


# In[7]:


def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        #FILL IN HERE
        output=model(data)
        criterion=nn.CrossEntropyLoss()
        loss=criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%100==0:
            print('Epoch:',epoch,',loss:',loss.item())


# In[5]:


# Q4d)


# In[14]:


def test(model,device,test_loader):
    model.eval()
    correct=0
    exampleSet=False
    example_data=numpy.zeros([10,28,28])
    example_pred=numpy.zeros(10)
    
    with torch.no_grad():
        for data, target in test_loader:
            data,target=data.to(device),target.to(device)
            #FILL IN HERE
            output=model(data)
            _, pred=torch.max(output.data,1)
            correct+=(pred==target).sum().item()
            
            if not exampleSet:
                for i in range(10):
                    example_data[i]=data[i][0].to("cpu").numpy()
                    example_pred[i]=pred[i].to("cpu").numpy()
                exampleSet=True
                
    print('Test set accuracy: ', 100.*correct/len(test_loader.dataset),'%')
    
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(example_data[i],cmap='gray',interpolation='none')
        plt.title(labels[example_pred[i]])
        plt.xticks([])
        plt.yticks([])
        plt.show()


# In[ ]:





# In[ ]:




