import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

rng = np.random.default_rng(1234)

fig, ax = plt.subplots(figsize = (9, 9))

traffic_data = list(map(int, pd.read_csv("./traffic-data.csv")["Traffic"]))
cluster_data = [15, 21, 12, 6, 20, 18, 20]
ax.scatter(traffic_data, cluster_data)
plt.xlabel("Traffic Amount")
plt.ylabel("Diversity Score")
plt.ylim((0, 50))
plt.title("Correlation of Traffic with Diversity Score")

b, a = np.polyfit(traffic_data, cluster_data, deg = 1)
xseq = np.linspace(0, 250, num=50)
ax.plot(xseq, a + b * xseq, color="k", lw=0.5)
plt.savefig("traffic_analysis.jpg")