from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Read and convert form inputs
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"])
            ]

            # Convert to NumPy array
            final_features = np.array(features).reshape(1, -1)

            # Predict
            result = model.predict(final_features)[0]

            flower_names = ["Setosa", "Versicolor", "Virginica"]
            prediction = flower_names[result]

        except ValueError:
            error = "Please enter valid numeric values only."

    return render_template("index.html",
                           prediction=prediction,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
