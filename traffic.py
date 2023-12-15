import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, weightedtau

rng = np.random.default_rng(1234)

fig, ax = plt.subplots(figsize = (9, 9))

traffic_data = list(map(float, pd.read_csv("./traffic-data.csv")["Traffic"]))
# cluster_data = [15, 21, 12, 6, 20, 18, 20]
# cluster_data = [2.5146040935886753, 2.6273013513690597, 2.1972792881524503, 1.6052071074554588, 2.7480040381780615, 2.6175834698506604, 2.728994705595429]
cluster_data = [0.09287796751353602, 0.09248242293131424, 0.1249323958896701, 0.2307692307692308, 0.07602339181286548, 0.0876390794086267, 0.07757963725041914]
classes = ['Doherty', 'Entropy', 'Gates', 'Low', 'Prima', 'Resnik', 'Wean']

ax.scatter(traffic_data, cluster_data)

for i, txt in enumerate(classes):
    ax.annotate(txt, (traffic_data[i], cluster_data[i]))

plt.xlabel("Traffic Amount")
plt.ylabel("Simpson Diversity Score")
# plt.ylim((0, 50))
# plt.ylim((0, 3.5))
plt.ylim((0, 0.3))
plt.title("Correlation of Traffic with Simpson Diversity Score")

b, a = np.polyfit(traffic_data, cluster_data, deg = 1)
xseq = np.linspace(0, 250, num=50)
ax.plot(xseq, a + b * xseq, color="k", lw=0.5)
plt.savefig("traffic_analysis.jpg")

yhat = (lambda x: a + (b * x))(np.array(traffic_data).astype(np.float64))
ybar = np.sum(cluster_data)/len(cluster_data)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((cluster_data - ybar)**2)
rsquared = ssreg/sstot

pearson, _ = pearsonr(traffic_data, cluster_data)
spearman, _ = spearmanr(traffic_data, cluster_data)
kendall = kendalltau(traffic_data, cluster_data).statistic
weighted = weightedtau(traffic_data, cluster_data).statistic

print(f"R^2: {rsquared}\nPearson: {pearson}\nSpearman: {spearman}"
      f"\nKendall: {kendall}\nWeighted: {weighted}")