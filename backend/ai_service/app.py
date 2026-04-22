from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import certifi
from datetime import datetime
from PIL import Image
from bson import ObjectId
import numpy as np

from utils import predict_image

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ✅ MongoDB Atlas Connection
try:
    client = MongoClient(
        os.getenv("MONGO_URI"),
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000
    )
    client.server_info()
    print("✅ MongoDB Atlas Connected Successfully")

except Exception as e:
    print("❌ MongoDB Connection Error:", e)
    client = None

# Database & Collection
if client is not None:
    db = client["crop_db"]
    collection = db["predictions"]
else:
    collection = None


# ✅ Helper: Clean ML output (handles numpy, objects)
def clean_result(result):
    clean = {}
    for k, v in result.items():
        if isinstance(v, (np.integer, np.floating)):
            clean[k] = float(v)
        elif isinstance(v, (list, tuple)):
            clean[k] = [float(i) if isinstance(i, (np.integer, np.floating)) else str(i) for i in v]
        else:
            clean[k] = str(v)
    return clean


# ✅ Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        # 🔥 Model prediction
        result = predict_image(image)

        # ✅ Clean ML output
        result = clean_result(result)

        # Add timestamp
        result["timestamp"] = str(datetime.now())

        # Save to MongoDB
        if collection is not None:
            inserted = collection.insert_one(result)
            result["_id"] = str(inserted.inserted_id)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ History API
@app.route("/history", methods=["GET"])
def history():
    try:
        if collection is None:
            return jsonify({"error": "Database not connected"}), 500

        data = []
        for doc in collection.find():
            doc["_id"] = str(doc["_id"])  # convert ObjectId → string
            data.append(doc)

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Root route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Disease API Running 🚀"})


# Run server
if __name__ == "__main__":
    app.run(port=5001 , debug=True)