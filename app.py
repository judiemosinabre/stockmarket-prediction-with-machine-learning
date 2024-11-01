# app.py

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('PSE.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Convert shorthand notation to numerical values in 'Vol.'
def convert_shorthand_to_num(value):
    if 'K' in value:
        return float(value.replace('K', '')) * 1e3
    elif 'M' in value:
        return float(value.replace('M', '')) * 1e6
    elif 'B' in value:
        return float(value.replace('B', '')) * 1e9
    else:
        return float(value)
df['Vol.'] = df['Vol.'].apply(convert_shorthand_to_num)

# Convert 'Change' column from percentages to floats
df['Change'] = df['Change'].str.replace('%', '').astype(float) / 100

# Prepare data
X = df[['Open', 'High', 'Low', 'Vol.', 'Change']].values
y = df['Price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Define routes
@app.route('/data')
def get_data():
    # Convert data to JSON format for Chart.js
    data = {
        "dates_train": df.index[:len(y_train)].strftime('%Y-%m-%d').tolist(),
        "y_train": y_train.tolist(),
        "y_pred_train": y_pred_train.tolist(),
        "dates_test": df.index[len(y_train):].strftime('%Y-%m-%d').tolist(),
        "y_test": y_test.tolist(),
        "y_pred_test": y_pred_test.tolist(),
    }
    return jsonify(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/psei')
def psei():
    return render_template('psei.html')

if __name__ == '__main__':
    app.run(debug=True)
