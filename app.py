from flask import Flask, render_template, request, send_from_directory, url_for, jsonify
import os
# Prevent HuggingFace transformers from importing TensorFlow unless explicitly needed.
# This avoids TensorFlow-related import errors on systems where TF is not installed or incompatible.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import torch
import sys
import types
# If we've disabled TensorFlow for transformers, inject a lightweight dummy module
# so that transformers' optional TF imports don't attempt to load a broken TF install.
if os.environ.get("TRANSFORMERS_NO_TF", "0") == "1":
    # Create a lightweight dummy tensorflow module with a valid __spec__ so
    # transformers' import checks (importlib.util.find_spec) don't raise errors.
    dummy_tf = types.ModuleType("tensorflow")
    try:
        import importlib.machinery
        dummy_tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
    except Exception:
        dummy_tf.__spec__ = None
    sys.modules.setdefault("tensorflow", dummy_tf)

from transformers import CLIPModel, CLIPTokenizer
from PIL import Image
import numpy as np
import cv2
import secrets
from PIL import Image
import os
import numpy as np
import cv2
import secrets

print("🚀 Starting Insulator Condition Detection Flask App...")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

print("📦 Loading CLIP model (This may take a few minutes on first run)...")
try:
    # Load pretrained CLIP model (no training needed)
    # Check if model exists locally first
    model_path = "./model_cache/models--openai--clip-vit-base-patch32"
    if os.path.exists(model_path):
        print("✅ Found cached model, loading from local cache...")
    
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir="./model_cache",
        local_files_only=os.path.exists(model_path),
    )
    print("✅ Model loaded successfully!")
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir="./model_cache",
        local_files_only=os.path.exists(model_path),
    )
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise

# -------------------------------
# Calibration: compute edge density threshold using sample folders
# -------------------------------
def compute_edge_density(path):
    try:
        img = Image.open(path).convert("L")
        img = img.resize((224, 224))
        arr = np.array(img)
        edges = cv2.Canny(arr, 100, 200)
        density = float(np.count_nonzero(edges)) / edges.size
        return density
    except Exception as e:
        print(f"Warning: failed to compute edge density for {path}: {e}")
        return 0.0

def calibrate_edge_threshold(healthy_dir="healthy", damaged_dir="damaged"):
    healthy_vals = []
    damaged_vals = []
    if os.path.isdir(healthy_dir):
        for fn in os.listdir(healthy_dir):
            fp = os.path.join(healthy_dir, fn)
            if os.path.isfile(fp):
                healthy_vals.append(compute_edge_density(fp))
    if os.path.isdir(damaged_dir):
        for fn in os.listdir(damaged_dir):
            fp = os.path.join(damaged_dir, fn)
            if os.path.isfile(fp):
                damaged_vals.append(compute_edge_density(fp))

    # use conservative default if no samples
    if len(healthy_vals) == 0 or len(damaged_vals) == 0:
        print("Calibration: insufficient sample images for calibration, using default threshold=0.03")
        return 0.03

    mean_h = float(np.mean(healthy_vals))
    mean_d = float(np.mean(damaged_vals))
    threshold = (mean_h + mean_d) / 2.0
    # expose means for normalization later
    global MEAN_HEALTHY, MEAN_DAMAGED
    MEAN_HEALTHY = mean_h
    MEAN_DAMAGED = mean_d
    print(f"Calibration complete: mean_healthy={mean_h:.4f}, mean_damaged={mean_d:.4f}, threshold={threshold:.4f}")
    return threshold

# Run calibration at startup
print("🔧 Running calibration using your healthy/ and damaged/ folders...")
EDGE_THRESHOLD = calibrate_edge_threshold()

# -------------------------------
# Optional: load a local Keras model (converted_keras)
# -------------------------------
# Default path provided by user (point to the H5 file inside converted_keras)
KERAS_MODEL_PATH = r"D:\insulator project_version2\insular project\converted_keras\keras_model.h5"
KERAS_MODEL = None
KERAS_AVAILABLE = False
# Only attempt to import TensorFlow and load the Keras model if the file exists.
if os.path.exists(KERAS_MODEL_PATH):
    print(f"📥 Keras model file found at {KERAS_MODEL_PATH}. Attempting to load TensorFlow and the model...")
    try:
        # Lazy import to avoid forcing TensorFlow import when not required or incompatible
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        KERAS_MODEL = load_model(KERAS_MODEL_PATH)
        KERAS_AVAILABLE = True
        print("✅ Keras model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to import TensorFlow or load Keras model: {e}")
        KERAS_MODEL = None
        KERAS_AVAILABLE = False
else:
    print(f"ℹ️ Keras model not found at {KERAS_MODEL_PATH}. Skipping Keras integration.")

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file uploaded."
        else:
            file = request.files['file']
            if file.filename != '':
                # Save with a secure filename
                file_extension = os.path.splitext(file.filename)[1]
                secure_filename = secrets.token_hex(16) + file_extension
                filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename)
                file.save(filename)
                
                # Create URL for the uploaded file
                image_url = url_for('uploaded_file', filename=secure_filename)

                # Load and preprocess image
                image = Image.open(filename).convert("RGB")
                image = image.resize((224, 224))  # CLIP standard input size
                
                # Enhance contrast and brightness slightly for better defect detection
                image_cv = np.array(image)
                image_cv = cv2.convertScaleAbs(image_cv, alpha=1.2, beta=10)
                image = Image.fromarray(image_cv)
                
                # Compute edge density heuristic for uploaded image
                edge_density = compute_edge_density(filename)
                print(f"🔎 Edge density: {edge_density:.4f} (threshold={EDGE_THRESHOLD:.4f})")

                # Default mode: ensemble (uses all available models)
                mode = request.form.get('mode', 'ensemble')

                # Run CLIP inference (always, for ensemble)
                texts = [
                    "a close-up, clean, undamaged, and intact electrical insulator with no cracks, burns, or dirt, working properly on a power line",
                    "a broken, cracked, dirty, burnt, or damaged electrical insulator showing signs of failure, wear, or malfunction"
                ]

                # Tokenize texts (no TensorFlow needed)
                text_inputs = tokenizer(texts, padding=True, return_tensors="pt")

                # Preprocess PIL image into CLIP pixel_values tensor
                def _preprocess_image_pil(img_pil):
                    arr = np.array(img_pil).astype('float32') / 255.0
                    # CLIP normalization parameters
                    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
                    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
                    arr = (arr - mean) / std
                    # HWC to CHW
                    arr = np.transpose(arr, (2, 0, 1))
                    tensor = torch.from_numpy(arr).unsqueeze(0)
                    return tensor

                pixel_values = _preprocess_image_pil(image)

                inputs = {
                    "input_ids": text_inputs["input_ids"],
                    "pixel_values": pixel_values,
                }
                if "attention_mask" in text_inputs:
                    inputs["attention_mask"] = text_inputs["attention_mask"]

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

                healthy_prob, damaged_prob = probs[0], probs[1]
                print(f"🔍 CLIP Model confidence — Healthy: {healthy_prob:.2f}, Damaged: {damaged_prob:.2f}")

                # Compute normalized edge score
                if 'MEAN_HEALTHY' in globals() and 'MEAN_DAMAGED' in globals() and MEAN_DAMAGED > MEAN_HEALTHY:
                    edge_score = (edge_density - MEAN_HEALTHY) / (MEAN_DAMAGED - MEAN_HEALTHY)
                    edge_score = max(0.0, min(1.0, edge_score))
                else:
                    edge_score = (edge_density - 0.0) / (0.2 - 0.0)
                    edge_score = max(0.0, min(1.0, edge_score))

                # Try to get Keras prediction
                keras_prob = None
                if KERAS_MODEL is not None:
                    try:
                        img_pil = Image.open(filename).convert('RGB')
                        try:
                            input_shape = KERAS_MODEL.input_shape
                        except Exception:
                            input_shape = None
                        if input_shape and len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
                            H = int(input_shape[1]); W = int(input_shape[2])
                        else:
                            H, W = 224, 224

                        img_resized = img_pil.resize((W, H))
                        arr = np.array(img_resized).astype('float32') / 255.0
                        if arr.ndim == 2:
                            arr = np.stack([arr, arr, arr], axis=-1)
                        arr_batch = np.expand_dims(arr, 0)
                        preds = KERAS_MODEL.predict(arr_batch, verbose=0)
                        if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
                            keras_prob = float(preds.ravel()[-1])
                        else:
                            p = np.array(preds)[0]
                            if p.sum() > 1e-6:
                                p = p / p.sum()
                            keras_prob = float(p[-1])
                        print(f"🤖 Keras model damaged_prob={keras_prob:.4f}")
                    except Exception as e:
                        print(f"⚠️ Error running Keras model: {e}")
                        keras_prob = None

                # Ensemble decision logic
                if mode == 'ensemble' and keras_prob is not None:
                    # Weighted ensemble: Keras (50%), CLIP (30%), Edge (20%)
                    ensemble_score = 0.5 * keras_prob + 0.3 * damaged_prob + 0.2 * edge_score
                    print(f"📊 Ensemble score: {ensemble_score:.4f} (Keras: {keras_prob:.2%}, CLIP: {damaged_prob:.2%}, Edge: {edge_score:.2%})")
                    
                    if ensemble_score > 0.5:
                        result = f"⚠️ Insulator is DAMAGED (Confidence: {ensemble_score:.2%})"
                    else:
                        result = f"✅ Insulator is HEALTHY (Confidence: {(1-ensemble_score):.2%})"
                
                elif mode == 'keras' and keras_prob is not None:
                    # Keras-only decision
                    if keras_prob > 0.5:
                        result = f"⚠️ Insulator is DAMAGED (Keras: {keras_prob:.2%})"
                    else:
                        result = f"✅ Insulator is HEALTHY (Keras: {keras_prob:.2%})"
                
                elif mode == 'clip':
                    # CLIP-only decision
                    confidence_margin = 0.10 + abs(healthy_prob - damaged_prob) * 0.1
                    if damaged_prob > healthy_prob + confidence_margin:
                        result = f"⚠️ Insulator is DAMAGED (CLIP: {damaged_prob:.2%})"
                    elif healthy_prob > damaged_prob + confidence_margin:
                        result = f"✅ Insulator is HEALTHY (CLIP: {healthy_prob:.2%})"
                    else:
                        # Use edge heuristic as tie-breaker
                        combined_score = 0.5 * damaged_prob + 0.5 * edge_score
                        if combined_score > 0.5:
                            result = f"⚠️ Insulator is DAMAGED (Combined: {combined_score:.2%})"
                        else:
                            result = f"✅ Insulator is HEALTHY (Combined: {combined_score:.2%})"
                
                else:
                    # Fallback to combined CLIP + Edge if Keras unavailable
                    combined_score = 0.6 * damaged_prob + 0.4 * edge_score
                    print(f"📊 Combined score: {combined_score:.4f} (CLIP: {damaged_prob:.2%}, Edge: {edge_score:.2%})")
                    if combined_score > 0.5:
                        result = f"⚠️ Insulator is DAMAGED (Confidence: {combined_score:.2%})"
                    else:
                        result = f"✅ Insulator is HEALTHY (Confidence: {(1-combined_score):.2%})"

    return render_template('index.html', result=result, filename=image_url if 'image_url' in locals() else None)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """JSON API endpoint for mobile apps"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded", "success": False}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename", "success": False}), 400
    
    try:
        # Save with a secure filename
        file_extension = os.path.splitext(file.filename)[1]
        secure_filename = secrets.token_hex(16) + file_extension
        filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename)
        file.save(filename)
        
        # Load and preprocess image
        image = Image.open(filename).convert("RGB")
        image = image.resize((224, 224))
        
        # Enhance contrast and brightness
        image_cv = np.array(image)
        image_cv = cv2.convertScaleAbs(image_cv, alpha=1.2, beta=10)
        image = Image.fromarray(image_cv)
        
        # Compute edge density
        edge_density = compute_edge_density(filename)
        print(f"🔎 Edge density: {edge_density:.4f}")

        # CLIP inference
        texts = [
            "a close-up, clean, undamaged, and intact electrical insulator with no cracks, burns, or dirt, working properly on a power line",
            "a broken, cracked, dirty, burnt, or damaged electrical insulator showing signs of failure, wear, or malfunction"
        ]

        text_inputs = tokenizer(texts, padding=True, return_tensors="pt")

        def _preprocess_image_pil(img_pil):
            arr = np.array(img_pil).astype('float32') / 255.0
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            arr = (arr - mean) / std
            arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr).unsqueeze(0)
            return tensor

        pixel_values = _preprocess_image_pil(image)

        inputs = {
            "input_ids": text_inputs["input_ids"],
            "pixel_values": pixel_values,
        }
        if "attention_mask" in text_inputs:
            inputs["attention_mask"] = text_inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

        healthy_prob, damaged_prob = probs[0], probs[1]
        print(f"🔍 CLIP — Healthy: {healthy_prob:.2f}, Damaged: {damaged_prob:.2f}")

        # Compute normalized edge score
        if 'MEAN_HEALTHY' in globals() and 'MEAN_DAMAGED' in globals() and MEAN_DAMAGED > MEAN_HEALTHY:
            edge_score = (edge_density - MEAN_HEALTHY) / (MEAN_DAMAGED - MEAN_HEALTHY)
            edge_score = max(0.0, min(1.0, edge_score))
        else:
            edge_score = (edge_density - 0.0) / (0.2 - 0.0)
            edge_score = max(0.0, min(1.0, edge_score))

        # Combined decision
        combined_score = 0.6 * damaged_prob + 0.4 * edge_score
        print(f"📊 Combined: {combined_score:.4f}")
        
        is_damaged = combined_score > 0.5
        confidence = combined_score if is_damaged else (1 - combined_score)
        
        response = {
            "success": True,
            "is_damaged": bool(is_damaged),
            "is_healthy": bool(not is_damaged),
            "confidence": float(confidence),
            "status": "Damaged" if is_damaged else "Healthy",
            "message": f"Insulator is {'DAMAGED' if is_damaged else 'HEALTHY'} - AI Powered Analysis",
            "details": {
                "clip_healthy": float(healthy_prob),
                "clip_damaged": float(damaged_prob),
                "edge_score": float(edge_score),
                "combined_score": float(combined_score)
            }
        }
        print(f"✅ API Response: {response}")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    print("🌐 Starting Flask server on host 0.0.0.0:5000 ...")
    try:
        # Explicitly set host to 0.0.0.0 to allow external connections
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        input("Press Enter to exit...")