import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers=Sequential(
            Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(6,6,kernel_size=3,stride=1,padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers=Sequential(
            Linear(10, 2)
        )

    def forward(self,x):
        x=self.cnn_layers(x)
        x=x.view(x.size(0),-1)
        x=self.linear_layers(x)
        return x

model=Net()
optimizer=Adam(model.parameters(),lr=0.07)
criterion=CrossEntropyLoss()
if torch.cuda.is_available():
    model=model.cuda()
    criterion=criterion.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch):
    model.train()
    tr_loss=0
    #get the training set
    x_train,y_train=Variable(train_x),Variable(train_y) #need to set up train_y

    #get validation set
    x_val,y_val=Variable(val_x),Variable(val_y) #need to set this up

    #set data types to use GPU if available
    if torch.cuda.is_available():
        x_train=x_train.cuda()
        y_train=y_train.cuda()
        x_val=x_val.cuda()
        y_val-y_val.cuda()

    #clearing gradients from model parameters
    optimizer.zero_grad()

    #prediction for training/validation set
    output_train=model(x_train)
    output_val=model(x_val)

    #Compute training and validation loss
    loss_train=criterion(output_train,y_train)
    loss_val=criterion(output_val,y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    #compute updated weights of model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss=loss_train.item()
    if epoch%2==0:
        #print validation loss if so
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

n_epochs=25
train_losses=[]
val_losses=[]

for epoch in range(n_epochs):
    train(epoch)