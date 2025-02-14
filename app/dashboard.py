# Step 1: Import Libraries
import dash
import dash_core_components as dcc
from dash import html  # ✅ Fixed import
import pandas as pd
import requests

# Step 2: Initialize Dash app
app = dash.Dash(__name__)

# Step 3: Fetch fraud data from the Flask API
def get_fraud_data():
    url = "http://127.0.0.1:5000/get_fraud_data"  # ✅ Ensure this matches your Flask API
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)

            # Debug: Print received columns
            print("✅ Data received from API. Columns:", df.columns.tolist())
            print(df.head())

            return df
        else:
            print(f"❌ Error fetching data. Status code: {response.status_code}")
            return pd.DataFrame()  # Return empty DataFrame in case of an error

    except Exception as e:
        print(f"❌ API request failed: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame in case of an exception

# Step 4: Define Dash layout
app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard"),
    dcc.Graph(id="fraud-trends"),
    dcc.Interval(id="interval-update", interval=60000, n_intervals=0)  # Auto-update every 60 sec
])

# Step 5: Callback to update fraud trends dynamically
@app.callback(
    dash.dependencies.Output("fraud-trends", "figure"),
    [dash.dependencies.Input("interval-update", "n_intervals")]
)
def update_graph(n):
    df = get_fraud_data()

    # Handle case when API returns an empty dataset
    if df.empty:
        print("⚠️ Warning: API returned an empty dataset.")
        return {"data": [], "layout": {"title": "No Data Available"}}

    # ✅ Updated column names based on API response
    required_columns = ["purchase_time", "class"]  # Adjusted column names
    for col in required_columns:
        if col not in df.columns:
            print(f"⚠️ Warning: Missing column '{col}' in dataset.")
            return {"data": [], "layout": {"title": f"Missing column '{col}'"}}

    # ✅ Plot fraud cases over time
    fig = {
        "data": [{"x": df["purchase_time"], "y": df["class"], "type": "line"}],
        "layout": {"title": "Fraud Cases Over Time"}
    }
    return fig

# Step 6: Run the Dashboard
if __name__ == "__main__":
    app.run_server(debug=True)
