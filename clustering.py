import feature_calc
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.pyplot as plt

groups = ['External', 'Cleaned', 'Doherty', 'Entropy', 'Gates', 'Low', 'Prima', 'Resnik', 'Wean']

num_seg_images = len(os.listdir('./seg-imgs'))
print(num_seg_images)
num_extern_images = len(os.listdir('./ext-seg-imgs'))
num_features = 16
random_seed = 42

filenames = []
images = np.empty(shape=(num_seg_images + num_extern_images, num_features))
group_labels = np.array([])
for fileidx, filename in enumerate(os.listdir('./seg-imgs')):
    filenames.append(filename)
    for idx, group in enumerate(groups):
        if (group in filename): 
            group_labels = np.append(group_labels, idx)
            break
    
    images[fileidx] = feature_calc.feature_calculation(f"./seg-imgs/{filename}")

for fileidx, filename in enumerate(os.listdir('./ext-seg-imgs')):
    filenames.append(filename)
    group_labels = np.append(group_labels, 0)
    images[fileidx + num_seg_images] = feature_calc.feature_calculation(f"./ext-seg-imgs/{filename}")

distortions = []
inertias = []
K = range(1, 100)

for k in K:
    kmeanModel = KMeans(n_clusters = k, init="k-means++", random_state=random_seed, n_init="auto").fit(images)
    distortions.append(sum(np.min(cdist(images, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / images.shape[0])
    inertias.append(kmeanModel.inertia_)

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.savefig("distortion.jpg")
plt.clf()


plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.savefig("inertia.jpg")
plt.clf()

print(images.shape)
print(group_labels.shape)

kmeans = KMeans(n_clusters=50, init="k-means++", random_state=42, n_init="auto").fit(images)
targets = kmeans.labels_

clusters = {idx: set() for idx in range(len(groups))}
for idx in range(images.shape[0]):
    clusters[group_labels[idx]].add(targets[idx])
print(list(map(lambda x: len(clusters[x]), clusters.keys())))

kmeans_plot = KMeans(n_clusters=10, init="k-means++", random_state=42, n_init="auto").fit(images)
targets_plot = kmeans_plot.labels_
reduced_images = PCA(2).fit_transform(images)
u_labels = np.unique(targets_plot)
for i in u_labels:
    plt.scatter(reduced_images[targets_plot == i, 0], reduced_images[targets_plot == i, 1], label = i)
plt.legend()
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')
plt.title('PCA Plot of Clusters (k = 10)')
plt.savefig("kmeans_plot.jpg")
plt.clf()