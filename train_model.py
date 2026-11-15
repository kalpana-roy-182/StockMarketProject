import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("stock_data.csv")

X = data[['Year']]
y = data['StockPrice']

# Polynomial Features
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Save model
joblib.dump(model, "stock_model.pkl")
joblib.dump(poly, "poly.pkl")

print("Model Trained Successfully")