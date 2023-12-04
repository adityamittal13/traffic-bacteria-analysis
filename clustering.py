import feature_calc
from sklearn.cluster import KMeans
import numpy as np
import os

groups = ['External', 'Cleaned', 'Doherty', 'Entropy', 'Gates', 'Low', 'Prima', 'Resnik', 'Wean']

num_seg_images = len(os.listdir('./seg-imgs'))
num_extern_images = len(os.listdir('./ext-seg-imgs'))

images = np.empty(shape=(num_seg_images + num_extern_images, 15))
group_labels = np.array([])
for fileidx, filename in enumerate(os.listdir('./seg-imgs')):
    for idx, group in enumerate(groups):
        if (group in filename): 
            group_labels = np.append(group_labels, idx)
            break
    
    images[fileidx] = feature_calc.feature_calculation(f"./seg-imgs/{filename}")

for fileidx, filename in enumerate(os.listdir('./ext-seg-imgs')):
    group_labels = np.append(group_labels, 0)
    images[fileidx + num_seg_images] = feature_calc.feature_calculation(f"./ext-seg-imgs/{filename}")

print(images.shape)
print(group_labels.shape)

kmeans = KMeans(n_clusters = 8, init="k-means++", random_state=0, n_init="auto").fit(images)
targets = kmeans.labels_

clusters = {idx: set() for idx in range(len(groups))}
for idx in range(images.shape[0]):
    clusters[group_labels[idx]].add(targets[idx])
print(clusters)