# Using Randomized Lasso
import pandas as pd
import numpy as np
from sklearn.linear_model import RandomizedLasso

dataset = pd.read_csv('ARJUNANADHI.csv')
X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, [6,9,12]].values
y_paddy=(Y[:,[0]]).ravel()
y_maize=Y[:,[1]].ravel()
y_cereals=Y[:,[2]].ravel()

rlasso = RandomizedLasso(alpha=0.04)
mapping = {0:'Meteorological', 1:'Hydrological',2:'Agricultural '}
fit=rlasso.fit(X,y_paddy)
Paddy=pd.DataFrame(fit.scores_)
Paddy.columns = ['Scores']
Paddy=Paddy.rename(mapping)
Paddy.plot.bar(title='Paddy',color='g',rot=0)

fit=rlasso.fit(X,y_maize)
Maize=pd.DataFrame(fit.scores_)
Maize.columns = ['Scores']
Maize=Maize.rename(mapping)
Maize.plot.bar(title='Maize',color='y',rot=0)

fit=rlasso.fit(X,y_cereals)
Cereals=pd.DataFrame(fit.scores_)
Cereals.columns = ['Scores']
Cereals=Cereals.rename(mapping)
Cereals.plot.bar(title='Cereals',color='c',rot=0)
