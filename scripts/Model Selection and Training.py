import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define a dictionary of models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

# Set an MLflow experiment name (this creates or uses an existing experiment)
mlflow.set_experiment("CreditCard_Fraud_Detection")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train_scaled, y_train)
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        # Calculate accuracy and print a classification report
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Log parameters and metrics to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, model_name)