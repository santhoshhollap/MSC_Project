'''
Usage: python3 TSNE.py
'''

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import seaborn as sns 

fts = np.load('featureMap.npy')
tsne = TSNE(n_components=2,perplexity=20.0, learning_rate= 5, n_iter=5000, verbose=1, random_state=123,square_distances=True)

z = tsne.fit_transform(fts)

x= []
y = []
for i,j in zip(z[:,0],z[:,1]):
    if (j < -50) and (i>-20) and (i<30):
        continue
    else:
        x.append(i)
        y.append(j)

print("len",len(z[:,0]))
plt.scatter( x, y)
plt.show()





