# LSTM model API

This repository includes a small Flask API that loads the LSTM model saved at `models/lstm/lstm_best.h5` and exposes a `/predict` endpoint.

Quick start (local development)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the API server:

```powershell
python api.py
```

3. Example request (single text):

```powershell
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text":"Free money!!!"}'
```

Notes and behavior
- The server attempts to load `models/lstm/lstm_tokenizer.json`. If it's missing or invalid it falls back to Keras' `hashing_trick` (no fitted tokenizer required). Predictions may differ from those obtained during training if the tokenizer does not match the one used for training.
- The server inspects the model input shape to infer the expected sequence length. If it cannot, it uses a default of 100.
- If you want exact behaviour matching training, ensure `models/lstm/lstm_tokenizer.json` contains the tokenizer JSON produced with `tokenizer.to_json()`.

If you'd like, I can:
- Add unit tests for the endpoint (fast, mocked model).
- Add a Dockerfile for production deployment.
- Try running a quick smoke test in this environment (may not be possible if TensorFlow isn't installed here).
