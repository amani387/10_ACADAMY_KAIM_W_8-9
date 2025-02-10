# Task 2 – Model Building and Training

## 1. Introduction

In this task, the goal was to build and train multiple machine learning models for fraud detection using two datasets (credit card transactions and fraud data). We compared different algorithms’ performance and set the stage for subsequent explainability, deployment, and dashboard tasks.

## 2. Technologies Used

- **Programming Language:** Python
- **Data Processing & Modeling:**  
  - **Pandas:** for data manipulation  
  - **scikit-learn:** for traditional models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) and data preparation (train-test split, scaling)
- **Deep Learning Framework:** TensorFlow/Keras  
  - Models built include: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), and Long Short-Term Memory (LSTM)
- **Experiment Tracking:** MLflow (for logging model parameters, metrics, and model versions)
- **Additional Tools:**  
  - **StandardScaler** for feature scaling  
  - **EarlyStopping** callback to prevent overfitting during neural network training

## 3. Data Preparation

Before model training, the following preprocessing steps were performed:

- **Feature & Target Separation:**  
  For the credit card dataset, the target variable is `Class` (1 indicates fraud, 0 indicates non-fraud).
- **Train-Test Split:**  
  An 80/20 split was used with stratification to preserve class imbalance.
- **Feature Scaling:**  
  Standardization was applied to the feature set.

### Data Preparation Code Snippet

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the cleaned credit card dataset
df_cc = pd.read_csv("cleaned_creditcard_data.csv")

# Ensure no missing values in the target column
df_cc = df_cc.dropna(subset=['Class'])

# Separate features and target
X = df_cc.drop('Class', axis=1)
y = df_cc['Class']

# Train-test split (stratified due to imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
