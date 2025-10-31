import json
import logging
from typing import List

from flask import Flask, request, jsonify

import numpy as np

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.preprocessing.text import hashing_trick
except Exception:
    load_model = None
    pad_sequences = None
    tokenizer_from_json = None
    hashing_trick = None


MODEL_PATH = "models/lstm/lstm_best.h5"
TOKENIZER_PATH = "models/lstm/lstm_tokenizer.json"

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_lstm_model(path: str):
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras is not available in this environment. Install tensorflow.")
    model = load_model(path)
    logger.info("Loaded model from %s", path)
    return model


def load_tokenizer(path: str):
    """Try to load a Keras Tokenizer from JSON. If not present or invalid, return None.

    The server will fallback to using hashing_trick which does not require a fitted tokenizer.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
            if not data:
                logger.warning("Tokenizer file %s is empty.", path)
                return None
            obj = json.loads(data)
        tokenizer = tokenizer_from_json(obj)
        logger.info("Loaded tokenizer from %s", path)
        return tokenizer
    except FileNotFoundError:
        logger.warning("Tokenizer file not found at %s. Falling back to hashing_trick.", path)
        return None
    except Exception as ex:
        logger.warning("Failed to load tokenizer (%s): %s. Falling back to hashing_trick.", path, ex)
        return None


def texts_to_sequences(texts: List[str], tokenizer, vocab_size: int):
    """Convert texts to sequences. If tokenizer is provided use it, otherwise use hashing_trick."""
    if tokenizer is not None:
        return tokenizer.texts_to_sequences(texts)
    if hashing_trick is None:
        raise RuntimeError("Keras preprocessing is not available. Install tensorflow.")
    return [hashing_trick(t, vocab_size, hash_function='md5', filters='') for t in texts]


def get_maxlen_from_model(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) >= 2 and shape[1] is not None:
            return int(shape[1])
    except Exception:
        pass
    # fallback
    return 100


_MODEL = None
_TOKENIZER = None
_VOCAB_SIZE = 20000
_MAXLEN = None


def initialize():
    global _MODEL, _TOKENIZER, _MAXLEN
    if _MODEL is None:
        _MODEL = load_lstm_model(MODEL_PATH)
        _TOKENIZER = load_tokenizer(TOKENIZER_PATH)
        _MAXLEN = get_maxlen_from_model(_MODEL)
        logger.info("Model expects sequence length: %s", _MAXLEN)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        initialize()
    except Exception as ex:
        logger.exception("Failed to initialize model: %s", ex)
        return jsonify(error=str(ex)), 500

    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify(error="Invalid or missing JSON body"), 400

    # Accept either {'text': '...'} or {'texts': ['...']}
    texts = []
    if isinstance(payload, dict) and "text" in payload:
        texts = [payload["text"]]
    elif isinstance(payload, dict) and "texts" in payload:
        texts = payload["texts"]
    else:
        return jsonify(error="JSON must contain 'text' or 'texts' key."), 400

    if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
        return jsonify(error="'text(s)' must be string or list of strings."), 400

    try:
        sequences = texts_to_sequences(texts, _TOKENIZER, _VOCAB_SIZE)
        seq_padded = pad_sequences(sequences, maxlen=_MAXLEN)
        preds = _MODEL.predict(seq_padded)
        # If binary classification, preds shape (n,1), else (n,num_classes)
        preds_list = preds.ravel().tolist() if preds.ndim == 2 and preds.shape[1] == 1 else preds.tolist()

        return jsonify(predictions=preds_list, shape=list(preds.shape)), 200
    except Exception as ex:
        logger.exception("Prediction failed: %s", ex)
        return jsonify(error=str(ex)), 500


if __name__ == "__main__":
    # Local dev server
    initialize()
    app.run(host="0.0.0.0", port=5000, debug=True)