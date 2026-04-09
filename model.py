import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utils import preprocess_data

# Load dataset
df = pd.read_csv("loan_data.csv")

# Preprocess
df = preprocess_data(df)

# Split data
target = df.columns[-1]
X = df.drop(target, axis=1)
y = df[target]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model trained and saved!")