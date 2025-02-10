import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import SimpleImputer

# Load the cleaned credit card dataset
df_cc = pd.read_csv(r"/content/cleaned_creditcard_data.csv")

# Separate features and target
X = df_cc.drop('Class', axis=1)
y = df_cc['Class']

# Handle NaN values in 'y' (if any)
y.fillna(y.mode()[0], inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute NaN values in X_train and X_test using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale the features after imputation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)  # Use imputed data
X_test_scaled = scaler.transform(X_test_imputed)      # Use imputed data