#Using Random Forest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('ARJUNANADHI.csv')
X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, [6,9,12]].values
y_paddy=(Y[:,[0]]).ravel()
y_maize=Y[:,[1]].ravel()
y_cereals=Y[:,[2]].ravel()

rf = RandomForestRegressor()
mapping = {0:'Meteorological', 1:'Hydrological',2:'Agricultural'}

rf.fit(X,y_paddy)
Paddy=pd.DataFrame(np.abs(rf.feature_importances_))
Paddy.columns = ['Importances']
Paddy=Paddy.rename(mapping)
Paddy.plot.bar(title='Paddy',color='g',rot=0)

rf.fit(X,y_maize)
Maize=pd.DataFrame(np.abs(rf.feature_importances_))
Maize.columns = ['Importances']
Maize=Maize.rename(mapping)
Maize.plot.bar(title='Maize',color='y',rot=0)

rf.fit(X,y_cereals)
Cereals=pd.DataFrame(np.abs(rf.feature_importances_))
Cereals.columns = ['Importances']
Cereals=Cereals.rename(mapping)
Cereals.plot.bar(title='Cereals',color='c',rot=0)

