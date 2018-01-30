import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Final_Output.csv')
dataset=dataset.dropna(axis=0, how='all')
paddy_dataset=dataset.iloc[:3, ]
maize_dataset=dataset.iloc[3:6, ]
cereals_dataset=dataset.iloc[6:9, ]

#Plotting the data
paddy_dataset = paddy_dataset.sort_values('Mean', ascending=False)
sns.factorplot(x="Mean", y="Features", data = paddy_dataset, kind="bar", 
               size=4, aspect=1.9, palette='coolwarm')
plt.title('Paddy')

maize_dataset = maize_dataset.sort_values('Mean', ascending=False)
sns.factorplot(x="Mean", y="Features", data = maize_dataset, kind="bar", 
               size=4, aspect=1.9, palette='coolwarm')
plt.title('Maize')

cereals_dataset = cereals_dataset.sort_values('Mean', ascending=False)
sns.factorplot(x="Mean", y="Features", data = cereals_dataset, kind="bar", 
               size=4, aspect=1.9, palette='coolwarm')
plt.title('Cereals')

