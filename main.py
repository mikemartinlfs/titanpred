import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch

# %%
# 1: Code each criteria, #2 Put each into a vector
def titanproc(filepath):
    df = pd.read_csv(filepath)

    try:
        survived = list(df['Survived'])
        survived2 = ["Y" if x == 1 else "N" for x in survived]
        survived = pd.get_dummies(survived2)
    except:
        survived = []

    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].copy()
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

    cabnum = [int(str(x)[1:]) / 1480 if len(str(x)) > 1 else 0 for x in
              cabin]  # Get and normalize the cab number, using 10*max when dividing to put the cabin number values in the hundredths place

    cabin = cabletter + cabnum  # The tenths place stands for the cabin letter, the hundredths through however far the values to stand for the cabin number

    normed = pd.DataFrame(list(zip(age, sex, sibsp, parch, fare, pclass, cabin, numcabs)),
                          columns=['Age', 'Sex', 'SibSP', 'Parch', 'Fare', 'Pclass', 'Cabin', '#Cabins'])
    return normed, survived


train, survived = titanproc('/Users/mikemartinlfs/Downloads/train.csv')
test, na = titanproc('/Users/mikemartinlfs/Downloads/test.csv')

# %%
# 3. Using a CNN, predict end result

X_train, X_test, y_train, y_test = train_test_split(train, survived, test_size=0.33, random_state=42)





