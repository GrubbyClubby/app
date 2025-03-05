from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

# Create the temp directory if it doesn't exist
os.makedirs('temp', exist_ok=True)

# Generate synthetic training data
X = np.array([[15, 60, 2], [20, 55, 3], [25, 50, 4], [30, 45, 5], [35, 40, 6]])
y = np.array([30, 40, 50, 60, 70])  # Heating demand values

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'temp/surrogate_model.pkl')

print("Model trained and saved successfully.")
