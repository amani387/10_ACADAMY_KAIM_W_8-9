# **Task 2 - Exploratory Data Analysis (EDA)**

This repository contains the code and steps for conducting **Exploratory Data Analysis (EDA)** as part of the 10 Academy AI Mastery Challenge. The analysis focuses on understanding the dataset, identifying patterns, and preparing data for feature engineering and modeling.

---

## **Steps for EDA**

### **1. Dataset Overview**

Load the dataset and inspect its structure:
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('path_to_dataset.csv')

# Display dataset details
print("Dataset Shape:", data.shape)
print(data.info())
print(data.head())
