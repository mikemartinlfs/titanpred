import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader,Dataset

class titanicdata(Dataset):
    def __init__(self,csvpath,mode='train'):
        self.mode=mode
        df=pd.read_csv(csvpath)
        if self.mode=='train':
            self.oup = list(df['Survived'])
        df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].copy()
        le=LabelEncoder()
        sex = list(pd.get_dummies(df['Sex'])['male'])
        pclass = list(df['Pclass'] / max(df['Pclass']))
        age = list(df['Age'] / max(df['Age']))
        agezero = -df['Age'].isna()
        age2 = []
        for i in range(len(agezero)):
            if agezero[i]:
                age2.append(age[i])
            else:
                age2.append(0)
        age = age2
        sibsp = list(df['SibSp'] / max(df['SibSp']))
        parch = list(df['Parch'] / max(df['Parch']))
        fare = list(df['Fare'] / max(df['Fare']))

        df['Cabin'] = df['Cabin'].fillna(0)
        numcabs = [(str(x).count(' ') + 1) / 4 if x != 0 else x - x for x in list(df['Cabin'])]

        emb = pd.get_dummies(df['Embarked'])
        embarked = list((emb['C'] + (2 * emb['Q']) + (3 * emb['S'])) / 3)
        cabin = [x.split()[0] if str(x).count(' ') > 0 else x for x in
                 list(df['Cabin'])]  # Get the first cabin listed if more than 1, otherwise return the cabin or 0
        firstcab = [str(x)[0] for x in cabin]  # Get the cabin letter - or 0 for nan values
        fc = pd.get_dummies(firstcab)  # convert to numeric values

        n = len(fc)
        cabletter = [0] * n
        weight = 0
        for i in fc:
            val = str(i)
            cabletter += fc[i] * weight
            weight += 1

        cabletter = cabletter / 10

        cabnum = [int(str(x)[1:]) / 148 if len(str(x)) > 1 else 0 for x in
                  cabin]
        normed = pd.DataFrame(list(zip(age, sex, sibsp, parch, fare, pclass, cabletter, cabnum, numcabs)))
        normed=normed.values.reshape(normed.shape[0],3,3)
        self.inp = normed

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        if self.mode=='train':
            inpt=torch.Tensor(self.inp)[idx]
            oupt=torch.Tensor(self.oup)[idx]
            return { 'inp': inpt,
                     'oup': oupt,}
        else:
            inpt=torch.Tensor(self.inp)[idx]
            return {'inp': inpt}

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

def train(epoch):
    model.train()
    tr_loss=0

    #get the training set
    x_train,y_train=Variable(train_x),Variable(train_y)

    #get validation set
    x_val,y_val=Variable(val_x),Variable(val_y)

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

data = titanicdata('/Users/mikemartinlfs/Downloads/train.csv')
n=len(data)
ntrain = int(n * 0.85)
nval = n - ntrain
train, val = torch.utils.data.random_split(data, [ntrain, nval])
train_data = DataLoader(dataset=train, batch_size=25, shuffle=True)
val_data = DataLoader(dataset=val, batch_size=25, shuffle=False)

batch=next(iter(train_data))
train_x=batch['inp']
train_y=batch['oup']

batchval=next(iter(val_data))
val_x=batchval['inp']
val_y=batchval['oup']

model=Net()
optimizer=Adam(model.parameters(),lr=0.07)
criterion=CrossEntropyLoss()
if torch.cuda.is_available():
    model=model.cuda()
    criterion=criterion.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs=100
train_losses=[]
val_losses=[]

for epoch in range(n_epochs):
    train(epoch)