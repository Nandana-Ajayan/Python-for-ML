## Supervised Learning Overview

**Definition:**  
Supervised learning is a core concept in machine learning where algorithms learn from labeled data, comprising input-output pairs, to predict outputs for unseen inputs.

**Key Components:**
- **Input Data:** Variables used for predictions, such as numerical or categorical features.
- **Output Labels:** Target variables for prediction, discrete (classification) or continuous (regression).
- **Training Data:** Labeled dataset used to train models, forming the basis for learning patterns.
- **Model:** Algorithm learning from training data to predict outcomes on new data.
- **Loss Function:** Measures difference between model predictions and actual labels, guiding optimization.
- **Optimization Algorithm:** Adjusts model parameters to minimize loss, often using gradient descent.
- **Evaluation Metrics:** Assess model performance on unseen data using accuracy, precision, etc.

**Types of Supervised Learning:**
- **Classification:** Predicts discrete categories like spam detection or sentiment analysis.
- **Regression:** Predicts continuous values such as house prices or temperature forecasting.

**Workflow:**
1. **Data Collection:** Gather labeled input-output pairs.
2. **Data Preprocessing:** Clean, handle missing values, and engineer features.
3. **Model Selection:** Choose an appropriate algorithm based on data and problem.
4. **Training:** Train the model to minimize loss on training data.
5. **Evaluation:** Assess model performance using validation or test datasets.
6. **Hyperparameter Tuning:** Optimize model performance further.
7. **Deployment:** Deploy model for real-world predictions.

**Applications:**
Supervised learning finds applications in:
- Email spam detection
- Handwritten digit recognition
- Medical diagnosis
- Autonomous driving
- Financial forecasting
- Natural language processing
- Recommender systems

In summary, supervised learning enables algorithms to predict outputs based on labeled data, driving advancements across diverse domains in artificial intelligence.
## Linar regression 

**Definition:**  
Supervised learning is a core concept in machine learning where algorithms learn from labeled data, comprising input-output pairs, to predict outputs for unseen inputs.

**Key Components:**
- **Input Data:** Variables used for predictions, such as numerical or categorical features.
- **Output Labels:** Target variables for prediction, discrete (classification) or continuous (regression).
- **Training Data:** Labeled dataset used to train models, forming the basis for learning patterns.
- **Model:** Algorithm learning from training data to predict outcomes on new data.
- **Loss Function:** Measures difference between model predictions and actual labels, guiding optimization.
- **Optimization Algorithm:** Adjusts model parameters to minimize loss, often using gradient descent.
- **Evaluation Metrics:** Assess model performance on unseen data using accuracy, precision, etc.

**Types of Supervised Learning:**
- **Classification:** Predicts discrete categories like spam detection or sentiment analysis.
- **Regression:** Predicts continuous values such as house prices or temperature forecasting.

**Workflow:**
1. **Data Collection:** Gather labeled input-output pairs.
2. **Data Preprocessing:** Clean, handle missing values, and engineer features.
3. **Model Selection:** Choose an appropriate algorithm based on data and problem.
4. **Training:** Train the model to minimize loss on training data.
5. **Evaluation:** Assess model performance using validation or test datasets.
6. **Hyperparameter Tuning:** Optimize model performance further.
7. **Deployment:** Deploy model for real-world predictions.

**Applications:**
Supervised learning finds applications in:
- Email spam detection
- Handwritten digit recognition
- Medical diagnosis
- Autonomous driving
- Financial forecasting
- Natural language processing
- Recommender systems

In summary, supervised learning enables algorithms to predict outputs based on labeled data, driving advancements across diverse domains in artificial intelligence.
sion

### Example problem:
***problem statement**
Analyse the California housing problem using linear rege=ression method
step-1:importinhg neve=cessary libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
```
step-2: Load the California Housing Dataset
```python
california_housing = fetch_california_housing()

# Display information about the dataset
``python
print(california_housing.DESCR)
```
Output:
Step 3: Prepare the data
```pyhon
X = california_housing.data
y = california_housing.target
```
Step 4: Split the data into training and testing sets
```pyhon
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Step 5: Create and train the linear regression model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
Step 5: Make predictions on the test set
```python
y_pred_linear = model.predict(X_test)
```
