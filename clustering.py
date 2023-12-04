import feature_calc
from sklearn.cluster import KMeans
import numpy as np
import os

groups = ['Cleaned', 'Doherty', 'Entropy', 'Gates', 'Low Traffic', 'Prima', 'Wean']

images = np.array([])
group_labels = np.array([])
for filename in os.listdir('./seg-imgs'):
    for idx, group in enumerate(groups):
        if (group in filename): 
            group_labels = np.append(group_labels, idx)
            break
    
    images = np.append(images, feature_calc.feature_calculation(f"./seg-imgs/{filename}"))

kmeans = KMeans(n_clusters = 8, init="k-means++", random_state=0, n_init="auto").fit(images)
targets = kmeans.labels_
    