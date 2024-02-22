## Use machine learning techniques to analyse the feedback of the Intel Unnati sessions
### Step-1:Importing necessary libraries 
The required libraries such as `pandas,numpy,seaborn,mathplotlib` etc are imported.
NB:Install the libraries if not present,using ``` pip install ``` corresponding library name.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
### Step-2:Loading the data to be analysed
The dataset can be in the form of csv file or may be directly from another sites.Specify the path of the dataset that we need to upload.
```python
#df_class=pd.read_csv("/content/survey_data.csv")
df_class=pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")

```
```python
df_class.head()
```
output:![1](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/915feae2-8169-4c0a-a4e7-bff6eaae44ed)
To make the table more attractive we can use the following commands:
```python
df_class.sample(5).style.set_properties(**{'background-color': 'darkgreen',
                           'color': 'white',
                           'border-color': 'darkblack'})
```
output:![bcolor data](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/211e64f8-cfc0-497b-9839-397e1dfb43ef)

### Step-3:Optimizing the dataset
Processing the data before tarining by removing the unnecessary columns.
```python
df_class.info()
```
In pandas, the `info()` method is used to get a concise summary of a DataFrame. This method provides information about the DataFrame, including the data types of each column, the number of non-null values, and memory usage. It's a handy tool for quickly assessing the structure and content of your DataFrame.

### Simple Breakdown of `info()` Method:

- **Index and Datatype of Each Column:** Shows the name of each column along with the data type of its elements (e.g., int64, float64, object).

- **Non-Null Count:** Indicates the number of non-null (non-missing) values in each column.

- **Memory Usage:** Provides an estimate of the memory usage of the DataFrame.

This method is especially useful when you want to check for missing values, understand the data types in your DataFrame, and get an overall sense of its size and composition. It's often used as a first step in exploring and understanding the characteristics of a dataset.
output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 174 entries, 0 to 173
Data columns (total 9 columns):
 #   Column                                                                                                                                     Non-Null Count  Dtype 
---  ------                                                                                                                                     --------------  ----- 
 0   Name of the Participant                                                                                                                    174 non-null    object
 1   Branch                                                                                                                                     174 non-null    object
 2   Semester                                                                                                                                   174 non-null    object
 3   Recourse Person of the session                                                                                                             174 non-null    object
 4   How would you rate the overall quality and relevance of the course content presented in this session?                                      174 non-null    int64 
 5   To what extent did you find the training methods and delivery style effective in helping you understand the concepts presented?            174 non-null    int64 
 6   How would you rate the resource person's knowledge and expertise in the subject matter covered during this session?                        174 non-null    int64 
 7   To what extent do you believe the content covered in this session is relevant and applicable to real-world industry scenarios?             174 non-null    int64 
 8   How would you rate the overall organization of the session, including time management, clarity of instructions, and interactive elements?  174 non-null    int64 
dtypes: int64(5), object(4)
memory usage: 12.4+ KB
### removing unnecessary columns
```python
df_class = df_class.drop(['Timestamp','Email ID','Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'],axis=1)
```
### specifying column names
```python
df_class.columns = ["Name","Branch","Semester","Resourse Person","Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
df_class.sample(5)
```
output:
### checking for null values and knowing the dimensions
```python
df_class.isnull().sum().sum()
```
Output:0
```python
# dimension
df_class.shape
```
Output:(174,9)
## Step-4: Exploratory Data Analysis
### creating an rp analysis in percentage
```python
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2)
```
explanation:
df_class["Resourse Person"]: This part extracts the column named "Resourse Person" from the DataFrame df_class.

.value_counts(): This function counts the occurrences of each unique value in the specified column, which is "Resourse Person" in this case.

normalize=True: The normalize parameter is set to True, which means the counts will be normalized to represent relative frequencies (percentages) instead of absolute counts.

*100: After normalization, the counts are multiplied by 100 to convert the relative frequencies into percentages.

round(..., 2): The resulting percentages are then rounded to two decimal places.
Output:
Resourse Person
Mrs. Akshara Sasidharan    34.48
Mrs. Veena A Kumar         31.03
Dr. Anju Pratap            17.24
Mrs. Gayathri J L          17.24
Name: proportion, dtype: float64
### creating a percentage analysis of Name-wise distribution of data
```python
round(df_class["Name"].value_counts(normalize=True)*100,2)
```
Output:
Name
Sidharth V Menon             4.02
Rizia Sara Prabin            4.02
Aaron James Koshy            3.45
Rahul Krishnan               3.45
Allen John Manoj             3.45
Christo Joseph Sajan         3.45
Jobinjoy Ponnappal           3.45
Varsha S Panicker            3.45
Nandana A                    3.45
Anjana Vinod                 3.45
Rahul Biju                   3.45
Kevin Kizhakekuttu Thomas    3.45
Lara Marium Jacob            3.45
Abia Abraham                 3.45
Shalin Ann Thomas            3.45
Abna Ev                      3.45
Aaron Thomas Blessen         2.87
Sebin Sebastian              2.87
Sani Anna Varghese           2.87
Bhagya Sureshkumar           2.87
Jobin Tom                    2.87
Leya Kurian                  2.87
Jobin Pius                   2.30
Aiswarya Arun                2.30
Muhamed Adil                 2.30
Marianna Martin              2.30
Anaswara Biju                2.30
Mathews Reji                 1.72
MATHEWS REJI                 1.72
Riya Sara Shibu              1.72
Riya Sara Shibu              1.72
Aiswarya Arun                1.15
Sarang kj                    1.15
Muhamed Adil                 1.15
Lisbeth Ajith                1.15
Jobin Tom                    0.57
Lisbeth                      0.57
Anaswara Biju                0.57
Aaron Thomas Blessen         0.57
Lisbeth Ajith                0.57
Marianna Martin              0.57
Name: proportion, dtype: float64

### Step-5:Visualization 
In this part,we are visualizing the analysed data part using graphs , pie charts etc.
```python
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Resourse Person', data=df_class)
#ax.bar_label(ax.containers[0])
plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
ax =plt.subplot(1,2,2)
ax=df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Resourse Person", fontsize = 20,color='Brown',pad=20);
```
### Create subplot with 1 row and 2 columns, selecting the first subplot
``ax = plt.subplot(1, 2, 1)``

### Create a count plot using Seaborn
``ax = sns.countplot(x='Resourse Person', data=df_class)``

### Set title for the first subplot
``plt.title("Faculty-wise distribution of data", fontsize=20, color='Brown', pad=20)``

### Move to the second subplot
``ax = plt.subplot(1, 2, 2)``

### Create a pie chart for the distribution of 'Resourse Person'
``ax = df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True)``

### Set title for the pie chart
``ax.set_title(label="Resourse Person", fontsize=20, color='Brown', pad=20)``
This code utilizes Matplotlib and Seaborn to generate a side-by-side visualization of the distribution of a categorical variable ("Resourse Person") in the DataFrame df_class. The first subplot displays a count plot (bar chart), while the second subplot presents a pie chart. Both charts provide insights into the frequency and proportion of different categories in the "Resourse Person" column.
Output:![pie chart](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/ba49a4c7-1f9c-4c1c-af33-5a36196b1cc0)
### Step-5:Creating a summary of responses
 A box and whisker plot or diagram (otherwise known as a boxplot), is a graph summarising a set of data. The shape of the boxplot shows how the data is distributed and it also shows any outliers. It is a useful way to compare different sets of data as you can draw more than one boxplot per graph.
 In this step we are creating box plot on various attributes and resource persons.
1)creating boxplot on content quality v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
Output:![rp,cq](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/3b8dc985-649d-4e0e-a527-43e545c307f7)

2)creating boxplot on Effectiveness v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
Outpu![rp ef](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/c8f13a2d-1ddf-4914-97a8-9f16ca3a8778)
t:
3)creating boxplot on Relevance v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
Output:![rp , re](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/51863a44-33fe-4f45-9963-76eb4323af07)
4)creating boxplot on Overall Organization v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
Output:![rp,oo](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/2dae80e1-181e-48b8-ba1e-93728d35b5b1)
5)creating boxplot on Branch  v/s Content quality

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
Output:![branch cq](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/4216d036-7835-4790-b0ce-91d20120755c)

## Step-6:Unsupervised machine learning
Using K-means Clustering to identify segmentation over student's satisfaction.
### Finding the best value of k using elbow method
# Elbow Method in Machine Learning

The elbow method is a technique used to determine the optimal number of clusters (k) in a clustering algorithm, such as k-means. It involves plotting the sum of squared distances (inertia) against different values of k and identifying the "elbow" point.

### Steps:

1. **Choose a Range of k Values:**
   - Select a range of potential values for the number of clusters.

2. **Run the Clustering Algorithm:**
   - Apply the clustering algorithm (e.g., k-means) for each value of k.
   - Calculate the sum of squared distances (inertia) for each clustering configuration.

3. **Plot the Elbow Curve:**
   - Plot the values of k against the corresponding sum of squared distances.
   - Look for an "elbow" point where the rate of decrease in inertia slows down.

4. **Identify the Elbow:**
   - The optimal k is often at the point where the inertia starts decreasing more slowly, forming an elbow.

### Interpretation:

- The elbow represents a trade-off between minimizing inertia and avoiding overfitting.
- It helps to find a balanced number of clusters for the given dataset.

Remember, while the elbow method is a useful heuristic, other factors like domain knowledge and analysis goals should also be considered in determining the final number of clusters.

```python
input_col=["Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
X=df_class[input_col].values
```
### Initialize an empty list to store the within-cluster sum of squares
```from sklearn.cluster import KMeans
wcss = []
```

### Try different values of k
```python

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# here inertia calculate sum of square distance in each cluster
```
### plotting sws v/s k value graphs
```python
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
Output:![Elbow method](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/d2cad2d9-c4d0-4369-8eb5-83a1e7667dda)


##  gridsearch method
Another method which can be used to find the optimized value of k is gridsearch method

```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```
```python
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
output:
Best Parameters: {'n_clusters': 5}
Best Score: -17.904781085966768
## Step-7:Implementing K-means Clustering
K-means Clustering is a model used in unsupervised learning.Here mean values are taken into account after fixing a centroid and the process is repeated.
```python
 Perform k-means clusteringprint("Best Parameters:", best_params)
print("Best Score:", best_score)
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)#
```
output:KMeans(n_clusters=3, n_init='auto', random_state=42)
## Extracting labels and cluster centers
Get the cluster labels and centroids
​```python
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
df_class.head()
```
Output:![extracting tables and cluster centers](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/bfee169b-2602-4e57-a86c-da5b58d205bd)

### Visualizing the clustering using first two features

```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
Output:
![clustering visualizastion](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/baa855ab-c4a1-47a8-943b-a8f8194e16b7)



