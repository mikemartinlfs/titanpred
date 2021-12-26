import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader,Dataset

# %%


class titanicdata(Dataset):
    def __init__(self,csvpath,mode='train'):
        self.mode=mode
        df=pd.read_csv(csvpath)
        if self.mode=='train':
            self.oup = list(df['Survived'])
            print(list(df['Survived']))
            print(self.oup)
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
        if self.mode=='train':
            inpt=torch.Tensor(self.inp)[idx]
            oupt=torch.Tensor(self.oup)[idx]
            return { 'inp': inpt,
                     'oup': oupt,}
        else:
            inpt=torch.Tensor(self.inp)[idx]
            return {'inp': inpt}

data=titanicdata('/Users/mikemartinlfs/Downloads/train.csv')
train_x=DataLoader(dataset=data,batch_size=64,shuffle=False)
dataiter=iter(train_x)
x,y=dataiter.next()
print(x)
print(y)
