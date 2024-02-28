# Technical Guide to Unsupervised Machine Learning

## Introduction to Unsupervised Learning

- Emphasize the fundamental difference between supervised and unsupervised learning, focusing on the absence of labeled data in the latter.
- Discuss the concept of learning from unlabeled data and the challenges it poses in terms of algorithmic design and evaluation.
- Provide concrete examples of unsupervised learning tasks, such as clustering to discover hidden structures in data or dimensionality reduction to compress information while preserving essential characteristics.

## Clustering

- Elaborate on the mathematical formulations and optimization objectives of clustering algorithms, such as minimizing intra-cluster distance and maximizing inter-cluster distance.
- Dive into the intricacies of different clustering techniques, like the initialization strategies in K-means or the hierarchical merging process in agglomerative hierarchical clustering.
- Explain how distance metrics, such as Euclidean distance or cosine similarity, are chosen based on the nature of the data and the desired clustering outcome.

## Dimensionality Reduction

- Provide a detailed explanation of dimensionality reduction methods, focusing on linear techniques like PCA and nonlinear methods like t-SNE.
- Discuss the mathematical underpinnings of PCA, including eigenvectors, eigenvalues, and covariance matrices, to elucidate how it captures the most significant variability in the data.
- Highlight the interpretability challenges associated with nonlinear techniques like t-SNE and the trade-offs between preserving local versus global structures in high-dimensional data visualization.

## Anomaly Detection

- Explore the probabilistic foundations of anomaly detection algorithms, such as Isolation Forest's use of random forests to isolate anomalies based on their rarity.
- Delve into the implementation details of One-Class SVM, including the selection of kernel functions and the optimization of hyperparameters like the kernel width.
- Discuss strategies for handling skewed class distributions and the impact of different anomaly detection thresholds on precision and recall metrics.

## Density Estimation

- Provide a rigorous treatment of density estimation techniques, such as Gaussian Mixture Models (GMM) and Kernel Density Estimation (KDE), from a probabilistic modeling perspective.
- Explain the Expectation-Maximization (EM) algorithm used to train GMMs iteratively, alternating between estimating cluster assignments and updating cluster parameters.
- Discuss the bandwidth selection problem in KDE and its implications for the smoothness of estimated density functions.

## Evaluation Metrics for Unsupervised Learning

- Deepen the understanding of evaluation metrics by discussing their mathematical formulations and intuitive interpretations.
- Provide examples of scenarios where different evaluation metrics excel, such as Silhouette Score for assessing cluster compactness and separation or Davies–Bouldin Index for measuring cluster dispersion.
- Highlight the importance of domain-specific knowledge in selecting appropriate evaluation metrics and interpreting their results effectively.

## Challenges and Best Practices

- Identify common challenges in unsupervised learning, such as the curse of dimensionality, scalability issues, and the lack of ground truth labels for validation.
- Offer practical strategies and best practices for addressing these challenges, such as feature scaling, data preprocessing techniques like normalization or standardization, and algorithmic hyperparameter tuning.
- Discuss the importance of exploratory data analysis (EDA) in understanding the inherent structure of the data and guiding the selection of appropriate unsupervised learning techniques.

By providing a deeper technical understanding of these concepts, practitioners can better leverage unsupervised machine learning techniques to extract meaningful insights from unlabeled data.

To perform Principal Component Analysis (PCA) using Python's scikit-learn library and visualize the results using matplotlib and seaborn the following codes will be useful

```python
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
```
```
df=pd.DataFrame([10,7,28,20,35],columns=["Marks"])
```
```
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df, method='ward'))
plt.axhline(y=3, color='r', linestyle='--')
```
#### Running clustering
```
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(df)
```

# Code for KMeans Clustering

#### 1.Importing Packages and Dataset
```
#Importing Libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline`
```
```
pip install seaborn
```
###### Importing the dataset
```
#Importing the dataset
iris=pd.read_csv("iris.csv")
```
```
iris.columns
```
```
#to show first five elements
iris.head()
```
```
iris.describe()
```
```
iris.info()
```
```
# Checking the null values in the dataset
iris.isnull().sum()
```
#### 2. Exploratory Data Analysis 

###### Box plot
```
# This shows how comparision of sepal length for different species
sns.boxplot(x = 'species', y='sepal_length', data = iris)
```
```
# This shows how comparision of sepal width for different species
sns.boxplot(x = 'species', y='sepal_width', data = iris)
```
```
# This shows how comparision of petal length for different species
sns.boxplot(x = 'species', y='petal_length', data = iris)
```
```
# This shows how comparision of petal width for different species
sns.boxplot(x = 'species', y='petal_width', data = iris)
```
###### Histogram
```
## Shows distribution of the variables
iris.hist(figsize=(8,6))
plt.show()
```
###### Pairplot
```
sns.pairplot(iris, hue='species')
```
```
iris.drop(['species'],axis = 1, inplace=True)
```

###### Correlation plot
```
figsize=[10,8]
plt.figure(figsize=figsize)
sns.heatmap(iris.corr(),annot=True)
plt.show()
```
#### 3. Finding Clusters with Elbow Method
```
ssw=[]
cluster_range=range(1,10)
for i in cluster_range:
    model=KMeans(n_clusters=i,init="k-means++",n_init=10, max_iter=300, random_state=0)
    model.fit(iris)
    ssw.append(model.inertia_)
```
```
ssw_df=pd.DataFrame({"no. of clusters":cluster_range,"SSW":ssw})
print(ssw_df)
```
```
plt.figure(figsize=(12,7))
plt.plot(cluster_range, ssw, marker = "o",color="cyan")
plt.xlabel("Number of clusters")
plt.ylabel("sum squared within")
plt.title("Elbow method to find optimal number of clusters")
plt.show()
```

#### 4. Building K Means model
```
# We'll continue our analysis with n_clusters=3
kmeans=KMeans(n_clusters=3, init="k-means++", n_init=10, random_state = 42)
# Fit the model
k_model=kmeans.fit(iris)
```
```
## It returns the cluster vectors i.e. showing observations belonging which clusters 
clusters=k_model.labels_
clusters
```
```
# Importing the dataset

iris=pd.read_csv("iris.csv")
```
```
iris['clusters']=clusters
print(iris.head())
print(iris.tail())
```
```
sns.boxplot(x = 'clusters', y='petal_width', data = iris)
```
```
sns.boxplot(x = 'clusters', y='petal_length', data = iris)
```
```
## Size of each cluster
iris['clusters'].value_counts()
```
```
# Centroid of each clusters
centroid_df = pd.DataFrame(k_model.cluster_centers_, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
centroid_df
```
```
### Visualizing the cluster based on each pair of columns

sns.pairplot(iris, hue='clusters')
```
