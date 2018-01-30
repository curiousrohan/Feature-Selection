# Feature Extraction with RFE
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('ARJUNANADHI.csv')
X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, [6,9,12]].values
y_paddy=(Y[:,[0]]).ravel()
y_maize=Y[:,[1]].ravel()
y_cereals=Y[:,[2]].ravel()

# feature extraction
model = LinearRegression()
rfe = RFE(model, n_features_to_select=1)
mapping = {0:'Meteorological', 1:'Hydrological',2:'Agricultural'}

fit = rfe.fit(X, y_paddy)
Paddy=pd.DataFrame(fit.ranking_)
Paddy.columns = ['Rank']
Paddy=Paddy.rename(mapping)
Paddy.plot.bar(title='Paddy' ,rot=0)

fit = rfe.fit(X, y_maize)
Maize=pd.DataFrame(fit.ranking_)
Maize.columns = ['Rank']
Maize=Maize.rename(mapping)
Maize.plot.bar(title='Maize', rot=0)

fit = rfe.fit(X, y_cereals)
Cereals=pd.DataFrame(fit.ranking_)
Cereals.columns = ['Rank']
Cereals=Cereals.rename(mapping)
Cereals.plot.bar(title='Cereals', rot=0)


#Findings:
'''In Paddy,Agricultural Drought affects the Yield most.
In Maize,Hydrological Drought affects the Yield most.
In Cereals,Hydrological Drought affects the Yield most.'''


