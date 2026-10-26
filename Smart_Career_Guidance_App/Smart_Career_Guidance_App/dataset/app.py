from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("model/career_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "Smart Career Guidance App is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)

    return jsonify({"Predicted Career": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)

