# build_model.py
# -----------------------------------
# This script trains a simple ML model
# and saves it as a .pkl file
# -----------------------------------

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

# Load built-in Iris dataset (no external CSV required)
iris = load_iris()
X = iris.data       # Features
y = iris.target     # Labels

# Train a simple model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl")
