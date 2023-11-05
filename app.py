from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
import json

app = Flask(__name__)
CORS(app)

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

@app.route("/")
def index():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)