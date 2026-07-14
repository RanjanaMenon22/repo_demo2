from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import numpy as np
import os
import threading
import webbrowser

MODEL_PATH = os.path.join("artifacts", "model.pkl")

app = Flask(__name__)
# Enable CORS (allow cross-origin requests)
CORS(app)
model = None
target_names = None

HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Iris Predictor</title>
  </head>
  <body>
    <h1>Iris Predictor</h1>
    <p>Enter 4 feature values (sepal length, sepal width, petal length, petal width) as comma-separated numbers:</p>
    <input id="features" size="40" value="5.1,3.5,1.4,0.2" />
    <button onclick="predict()">Predict</button>
    <pre id="result"></pre>

    <script>
      async function predict() {
        const raw = document.getElementById('features').value;
        const arr = raw.split(',').map(x => parseFloat(x.trim()));
        const resEl = document.getElementById('result');
        resEl.textContent = 'Calling server...';
        try {
          const resp = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: arr })
          });
          const j = await resp.json();
          resEl.textContent = JSON.stringify(j, null, 2);
        } catch (err) {
          resEl.textContent = 'Error: ' + err;
        }
      }
    </script>
  </body>
</html>
"""


def load_model():
    global model, target_names
    if model is not None:
        return model, target_names
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py to create it.")
    model = joblib.load(MODEL_PATH)
    # try to get human-readable class names
    try:
        from sklearn.datasets import load_iris
        target_names = load_iris().target_names.tolist()
    except Exception:
        # fallback to numeric class labels as strings
        target_names = [str(c) for c in getattr(model, 'classes_', range(0))]
    return model, target_names


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or 'input' not in data:
        return jsonify({'error': 'Request must be JSON with an "input" field containing a list of features.'}), 400
    features = data['input']

    # Strict validation: require exactly 4 numeric features
    if not isinstance(features, (list, tuple)):
        return jsonify({'error': 'Input must be a list of numeric features.'}), 400
    if len(features) != 4:
        return jsonify({'error': 'Exactly 4 features required: [sepal_length, sepal_width, petal_length, petal_width]'}), 400
    try:
        X = np.array([float(x) for x in features], dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({'error': f'All features must be numeric: {e}'}), 400

    try:
        model, names = load_model()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    try:
        pred = model.predict(X)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X).tolist()
        numeric = int(pred[0]) if hasattr(pred, '__iter__') else int(pred)
        label = names[numeric] if numeric < len(names) else str(numeric)
        return jsonify({'prediction': numeric, 'label': label, 'proba': proba})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health endpoint returning basic status and whether model loads."""
    ok = True
    model_loaded = False
    msg = ''
    try:
        load_model()
        model_loaded = True
    except Exception as e:
        ok = False
        msg = str(e)
    return jsonify({'ok': ok, 'model_loaded': model_loaded, 'message': msg})


if __name__ == '__main__':
    # Host and port can be configured via environment variables:
    # APP_HOST (default: 0.0.0.0) and PORT (default: 5000)
    host = os.getenv('APP_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))

    # Optionally auto-open the browser. Set AUTO_OPEN to '0' or 'false' to disable.
    auto_open = os.getenv('AUTO_OPEN', '1').lower() not in ('0', 'false', 'no')
    url = f'http://127.0.0.1:{port}/'
    if auto_open:
        try:
            # Open the local loopback URL so it works even when binding to 0.0.0.0
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    # Run the Flask app on the configured host and port
    app.run(host=host, port=port)
