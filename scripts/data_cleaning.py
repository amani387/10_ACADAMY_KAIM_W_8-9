import pandas as pd
import gdown
import ipaddress
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# STEP 1: Download the Fraud Dataset from Google Drive
# ============================================================

# The Google Drive shared file ID
file_id = "1Z5eXtQHqHksvQ7o8zYGtpfdYyr0VItfK"
download_url = f"https://drive.google.com/uc?id={file_id}"
output_file = "Fraud_Data.csv"

# Download the file (if it does not already exist)
gdown.download(download_url, output_file, quiet=False)

# ============================================================
# STEP 2: Load the Datasets
# ============================================================

# Load Fraud_Data, Credit Card Data, and IP-to-Country data
fraud_df = pd.read_csv("Fraud_Data.csv")
creditcard_df = pd.read_csv("creditcard1.csv")
ip_country_df = pd.read_csv("IpAddress_to_Country.csv")

# ============================================================
# STEP 3: Data Cleaning - IP Address to Country Data
# ============================================================

# Convert lower_bound_ip_address to integer (it was a float)
ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(int)

# ============================================================
# STEP 4: Data Cleaning - Fraud Data
# ============================================================

# Remove duplicate rows from fraud_df (if any)
fraud_df.drop_duplicates(inplace=True)

# Check for missing values and print a summary
print("Missing values in Fraud Data:")
print(fraud_df.isna().sum())

# Convert timestamp columns to datetime objects.
# (Assuming columns are named 'signup_time' and 'purchase_time')
fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'], errors='coerce')
fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'], errors='coerce')

# ============================================================
# STEP 5: Convert IP Address Strings to Integer
# ============================================================

# Define a function that converts an IPv4 address (string) to an integer.
def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except Exception:
        return np.nan

# Apply the conversion to create a new column 'ip_int'
fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)

# ============================================================
# STEP 6: Merge Fraud Data with IP-to-Country Data
# ============================================================

# Define a function that maps an IP (as an integer) to a country using the IP ranges
def map_ip_to_country(ip_int):
    # Find the row in ip_country_df where ip_int falls between the lower and upper bounds
    row = ip_country_df[(ip_country_df['lower_bound_ip_address'] <= ip_int) & 
                        (ip_country_df['upper_bound_ip_address'] >= ip_int)]
    if not row.empty:
        return row.iloc[0]['country']
    else:
        return np.nan

# Create a new column 'country' in fraud_df by applying the mapping function
fraud_df['country'] = fraud_df['ip_int'].apply(map_ip_to_country)

# ============================================================
# STEP 7: Feature Engineering - Time-Based Features
# ============================================================

# Extract the hour of day and day of week from the purchase_time column
fraud_df['purchase_hour'] = fraud_df['purchase_time'].dt.hour
fraud_df['purchase_dayofweek'] = fraud_df['purchase_time'].dt.dayofweek

# ============================================================
# STEP 8: Normalize a Key Feature (purchase_value)
# ============================================================

# Using MinMaxScaler to normalize the 'purchase_value' column (assumed to be the purchase amount)
scaler = MinMaxScaler()
# Reshape is required because scaler expects a 2D array
fraud_df['purchase_value_scaled'] = scaler.fit_transform(fraud_df[['purchase_value']])

# ============================================================
# STEP 9: Exploratory Data Analysis (EDA) - Quick Look
# ============================================================

# Print a summary of the fraud dataset to inspect data types and new columns
print("\nFraud Data Info:")
print(fraud_df.info())

# Display the first few rows to verify the changes
print("\nFraud Data Sample:")
print(fraud_df.head())

# Optionally, you can also print summary statistics
print("\nFraud Data Summary Statistics:")
print(fraud_df.describe())

# ============================================================
# STEP 10: (Optional) Quick EDA on Credit Card Data
# ============================================================

print("\nCredit Card Data Info:")
print(creditcard_df.info())
print("\nCredit Card Data Sample:")
print(creditcard_df.head())
