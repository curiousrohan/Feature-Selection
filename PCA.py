import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Importing the dataset
dataset = pd.read_csv('ARJUNANADHI.csv')
X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, [5,7,9]].values

y_paddy=Y[:,[0]]
y_paddy=pd.DataFrame(data=y_paddy, columns=['YIELD(PADDY)'])

y_maize=Y[:,[1]]
y_maize=pd.DataFrame(data=y_maize, columns=['YIELD(MAIZE)'])

y_cereals=Y[:,[2]]
y_cereals=pd.DataFrame(data=y_cereals, columns=['YIELD(CEREALS)'])

#Choosing number of Principal Components
pca = PCA(n_components = None)
reducedX= pca.fit_transform(X)
totalvariance=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
totalvariance=np.insert(totalvariance, 0, 0)
plt.plot(totalvariance,linewidth=2.0)
plt.xlabel('Dimensions')
plt.ylabel('Percentage of Variance retained')

# Applying PCA
pca = PCA(n_components = 2)
reducedX= pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
variance_retained=np.sum(pca.explained_variance_ratio_)*100
print(pca.explained_variance_)


PCA1_meteorContribution=(pca.components_[0, 0])*100
PCA1_hydroContribution=(pca.components_[0, 1])*100
PCA1_agriContribution=(pca.components_[0, 2])*100

PCA2_meteorContribution=pca.components_[1, 0]*100
PCA2_hydroContribution=pca.components_[1, 1]*100
PCA2_agriContribution=pca.components_[1, 2]*100


