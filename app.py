from flask import Flask, request, jsonify
import joblib
import os
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("artifacts/model.pkl")

if not MODEL_PATH.exists():
    # convenience: train if model missing
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)

# ── Browser UI ────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris Flower Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 500px; margin: 60px auto; padding: 20px; }
            h2   { color: #333; }
            label { display: block; margin-top: 14px; font-weight: bold; }
            input { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; font-size: 15px; }
            button { margin-top: 20px; padding: 10px 30px; background: #4CAF50; color: white;
                     border: none; font-size: 16px; cursor: pointer; border-radius: 4px; }
            button:hover { background: #45a049; }
            #result { margin-top: 24px; padding: 16px; border-radius: 6px; font-size: 18px; display: none; }
            .success { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
            .error   { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
        </style>
    </head>
    <body>
        <h2>🌸 Iris Flower Predictor</h2>
        <p>Enter the 4 measurements and click Predict.</p>

        <label>Sepal Length (cm)</label>
        <input type="number" id="sl" placeholder="e.g. 5.1" step="0.1">

        <label>Sepal Width (cm)</label>
        <input type="number" id="sw" placeholder="e.g. 3.5" step="0.1">

        <label>Petal Length (cm)</label>
        <input type="number" id="pl" placeholder="e.g. 1.4" step="0.1">

        <label>Petal Width (cm)</label>
        <input type="number" id="pw" placeholder="e.g. 0.2" step="0.1">

        <button onclick="predict()">Predict</button>

        <div id="result"></div>

        <script>
            const flowerNames = { 0: "Setosa 🌼", 1: "Versicolor 🌸", 2: "Virginica 🌺" };

            async function predict() {
                const features = [
                    parseFloat(document.getElementById("sl").value),
                    parseFloat(document.getElementById("sw").value),
                    parseFloat(document.getElementById("pl").value),
                    parseFloat(document.getElementById("pw").value)
                ];

                const resultDiv = document.getElementById("result");

                if (features.some(isNaN)) {
                    resultDiv.className = "error";
                    resultDiv.innerText = "Please fill in all 4 values.";
                    resultDiv.style.display = "block";
                    return;
                }

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ features: features })
                });

                const data = await response.json();

                if (data.prediction !== undefined) {
                    resultDiv.className = "success";
                    resultDiv.innerHTML = "Prediction: <strong>" + flowerNames[data.prediction] + "</strong>";
                } else {
                    resultDiv.className = "error";
                    resultDiv.innerText = "Error: " + data.error;
                }
                resultDiv.style.display = "block";
            }
        </script>
    </body>
    </html>
    """

# ── Health check ──────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ── Prediction API ────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "send JSON with key 'features'"}), 400
    features = data["features"]
    try:
        pred = model.predict([features])
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)