# Customer segmentation system
👋 Welcome to my E-Commerce Customer Segmentation project! In this project, I have used K-Means Clustering algorithm to cluster the customers based on their activity on the e-commerce site. The aim of this project is to group the customers based on their shared characteristics that distinguish them from other users.

# Dataset
The dataset used for this project is available [here](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

# Model
In this project, K-Means Clustering algorithm has been used to cluster customers based on their activity on an e-commerce site. The aim of this project is to segment customers based on their shared characteristics that distinguish them from other users. The dataset contains 36 columns, where the first column is the unique numbering of customers, the second column is the gender of the customer, and the remaining 35 columns (brands) contain the number of times customers have searched them
~~~
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
SSE = []
for cluster in range(1,10):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(scaled_features)
    SSE.append(kmeans.inertia_)
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
~~~

# Results
The models achieved an accuracy of 95% on the test set for age prediction and 90% for gender prediction. 

# Acknowledgements
This project was inspired by [here](https://www.kaggle.com/code/fabiendaniel/customer-segmentation)
