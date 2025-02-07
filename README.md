# Fraud Detection Challenge: E-commerce & Credit Card Transactions

Welcome to this week's challenge! In this project, build an end-to-end fraud detection pipeline using machine learning.  will work with three datasets, perform data analysis and preprocessing, build and explain multiple machine learning models, deploy  solutions as a REST API using Flask (and Docker), and create an interactive dashboard with Dash.

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Learning Outcomes](#learning-outcomes)
- [Tasks and Deliverables](#tasks-and-deliverables)
  - [Task 1: Data Analysis and Preprocessing](#task-1---data-analysis-and-preprocessing)
  - [Task 2: Model Building and Training](#task-2---model-building-and-training)
  - [Task 3: Model Explainability](#task-3---model-explainability)
  - [Task 4: Model Deployment and API Development](#task-4---model-deployment-and-api-development)
  - [Task 5: Dashboard Development with Flask and Dash](#task-5---dashboard-development-with-flask-and-dash)
- [Timeline & Key Dates](#timeline--key-dates)
- [Tutorial Schedule](#tutorial-schedule)
- [Team & Tutors](#team--tutors)
- [Setup and Running Instructions](#setup-and-running-instructions)
- [Additional Information](#additional-information)

---

## Overview

In this challenge, you will create a complete fraud detection solution by:

- **Analyzing and preprocessing data** from e-commerce and bank transaction datasets.
- **Building and comparing several machine learning models** including Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, MLP, CNN, RNN, and LSTM.
- **Explaining model predictions** using SHAP and LIME for enhanced transparency.
- **Deploying your model** as a REST API with Flask and containerizing the solution with Docker.
- **Developing an interactive dashboard** with Dash to visualize insights from your data.

---

## Datasets

### 1. Fraud_Data.csv
- **user_id:** Unique identifier for the user.
- **signup_time:** Timestamp when the user signed up.
- **purchase_time:** Timestamp of the purchase.
- **purchase_value:** Purchase value in dollars.
- **device_id:** Unique identifier for the device.
- **source:** Source of user traffic (e.g., SEO, Ads).
- **browser:** Browser used for the transaction.
- **sex:** Gender of the user (M for male, F for female).
- **age:** Age of the user.
- **ip_address:** IP address from which the transaction was made.
- **class:** Target variable (1 for fraudulent, 0 for non-fraudulent).

### 2. IpAddress_to_Country.csv
- **lower_bound_ip_address:** Lower bound of the IP address range.
- **upper_bound_ip_address:** Upper bound of the IP address range.
- **country:** Country corresponding to the IP address range.

### 3. creditcard.csv
- **Time:** Seconds elapsed between this transaction and the first transaction.
- **V1 to V28:** Anonymized PCA-derived features.
- **Amount:** Transaction amount in dollars.
- **Class:** Target variable (1 for fraudulent, 0 for non-fraudulent).

---

## Learning Outcomes

### Skills
- Deploying machine learning models using Flask.
- Containerizing applications with Docker.
- Creating REST APIs for ML models.
- Testing and validating APIs.
- Developing end-to-end deployment pipelines.
- Building scalable and portable ML solutions.
- Designing interactive dashboards with Dash.

### Knowledge
- Principles of model deployment and serving.
- Best practices for REST API development.
- Containerization benefits and techniques.
- Real-time prediction serving strategies.
- Security practices in API development.
- Monitoring and maintaining deployed models.

### Communication
- Reporting on statistically complex issues to stakeholders.

### Competency Mapping
This challenge will help you build competencies in:
- Global-level professionalism.
- Articulating business value.
- Collaboration and stakeholder communication.
- Software development frameworks and CI/CD with GitHub.
- Advanced Python programming (Pandas, Matplotlib, NumPy, Scikit-learn, Prophet, etc.).
- SQL programming (MySQL: create, read, write operations).
- Data and analytics engineering (data filtering, transformation, and warehouse management).
- Model explainability (using SHAP and LIME).
- API development (using Flask).
- Dashboard development (using Dash).

---

## Tasks and Deliverables

### Task 1 – Data Analysis and Preprocessing
- **Handle Missing Values:** Impute or drop missing data.
- **Data Cleaning:** Remove duplicates and correct data types.
- **Exploratory Data Analysis (EDA):**
  - Univariate analysis
  - Bivariate analysis
- **Merge Datasets for Geolocation Analysis:**
  - Convert IP addresses to integer format.
  - Merge `Fraud_Data.csv` with `IpAddress_to_Country.csv`.
- **Feature Engineering:**
  - Create transaction frequency and velocity features.
  - Derive time-based features (e.g., `hour_of_day`, `day_of_week`).
- **Normalization and Scaling:** Encode categorical features.

### Task 2 – Model Building and Training
- **Data Preparation:** Separate features and target (`Class` for creditcard.csv and `class` for Fraud_Data.csv) and perform a train-test split.
- **Model Selection:** Compare multiple models including:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
- **Training and Evaluation:** Train the models on both datasets.
- **MLOps:** Use MLflow for experiment tracking, logging parameters, and model versioning.

### Task 3 – Model Explainability
- **SHAP (Shapley Additive Explanations):**
  - Install SHAP: `pip install shap`
  - Generate SHAP summary, force, and dependence plots.
- **LIME (Local Interpretable Model-Agnostic Explanations):**
  - Install LIME: `pip install lime`
  - Generate feature importance plots for individual predictions.

### Task 4 – Model Deployment and API Development
- **Flask API Setup:**
  - Create a Flask application (e.g., `serve_model.py`) to serve your model.
  - Define REST API endpoints for predictions and model information.
  - Integrate logging using Flask-Logging for monitoring requests and errors.
- **Dockerize the Application:**
  - Create a `Dockerfile` to containerize the Flask API:
    ```dockerfile
    # Use an official Python runtime as a parent image
    FROM python:3.8-slim

    # Set the working directory in the container
    WORKDIR /app

    # Copy the current directory contents into the container at /app
    COPY . .

    # Install dependencies
    RUN pip install -r requirements.txt

    # Expose port 5000 for the API
    EXPOSE 5000

    # Run the Flask application
    CMD ["python", "serve_model.py"]
    ```
  - Build and run the Docker container:
    ```bash
    docker build -t fraud-detection-model .
    docker run -p 5000:5000 fraud-detection-model
    ```

### Task 5 – Dashboard Development with Flask and Dash
- **Create a Flask Endpoint:**
  - Develop an endpoint that reads fraud data from a CSV file and serves summary statistics and trends.
- **Design the Dashboard with Dash:**
  - Build interactive visualizations including:
    - Total transactions, fraud cases, and fraud percentages.
    - A line chart showing fraud trends over time.
    - Geographic analysis of fraud occurrences.
    - Comparative bar charts for fraud cases across devices and browsers.

---

## Setup and Running Instructions

### Prerequisites
- **Python 3.8+**
- **Docker** (for containerization)
- Required Python libraries: Pandas, NumPy, Scikit-learn, Matplotlib, SHAP, LIME, Flask, Dash, MLflow, etc.


