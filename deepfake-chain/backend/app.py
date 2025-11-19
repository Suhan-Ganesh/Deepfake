from flask import Flask, request, jsonify
from flask_cors import CORS
from blockchain_connect import upload_to_blockchain, view_blockchain
from deepfake_detector import get_detector
import hashlib
import mimetypes
import os

# Global detector instance
# Set force_fallback=True if the model is giving incorrect predictions
# To use multiple models, provide model_paths parameter with a list of model file paths
# Example: detector = get_detector(model_paths=['models/Meso4_DF.h5', 'models/Meso4_F2F.h5'])

# Auto-detect available models
def detect_model_paths():
    """Auto-detect available model files"""
    model_paths = []
    
    # Check models directory
    if os.path.exists("models") and os.path.isdir("models"):
        for f in os.listdir("models"):
            if f.endswith('.h5'):
                model_paths.append(os.path.join("models", f))
    
    # Check current directory for MesoNet models
    for f in os.listdir("."):
        if f.endswith('.h5') and f.startswith('mesonet'):
            model_paths.append(f)
    
    # Check for PyTorch models
    if os.path.exists("models") and os.path.isdir("models"):
        for f in os.listdir("models"):
            if f.endswith('.pt') or f.endswith('.pth'):
                model_paths.append(os.path.join("models", f))
    
    return model_paths if model_paths else None

# Initialize detector with all available models
model_paths = detect_model_paths()
# Removed print statement to prevent console output

# Create detector
detector = get_detector(force_fallback=False, model_paths=model_paths)

# If no trained models are available, provide guidance
if not detector.model_loaded:
    # Removed print statements to prevent console output
    pass
else:
    # Removed print statement to prevent console output
    # Set optimal parameters for trained models
    detector.set_detection_parameters(
        threshold=0.5,
        min_agreement=0.7,
        confidence_boost=1.3
    )

app = Flask(__name__)
CORS(app)

# ✅ Root route (test connection)
@app.route('/')
def home():
    return jsonify({"status": "Backend running successfully 🚀"}), 200


# ✅ Upload image/video to blockchain with deepfake detection
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read file bytes
        file_data = file.read()
        
        # Generate SHA-256 hash
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        # Determine file type
        file_type = 'image'
        if file.filename:
            mime_type, _ = mimetypes.guess_type(file.filename)
            if mime_type and mime_type.startswith('video'):
                file_type = 'video'
        
        # Perform deepfake detection
        is_deepfake, confidence, detection_method = detector.detect(file_data, file_type)
        
        # Removed print statements to prevent console output
        
        # Prepare blockchain data
        blockchain_data = {
            'hash': file_hash,
            'filename': file.filename,
            'file_type': file_type,
            'is_deepfake': is_deepfake,
            'confidence': round(confidence, 4),
            'detection_method': detection_method
        }

        # Upload to blockchain
        tx_hash = upload_to_blockchain(file_data, blockchain_data)

        return jsonify({
            "message": "File successfully analyzed and uploaded to blockchain",
            "transaction_hash": tx_hash,
            "file_hash": file_hash,
            "filename": file.filename,
            "file_type": file_type,
            "is_deepfake": is_deepfake,
            "confidence": round(confidence, 4),
            "detection_method": detection_method
        }), 200

    except Exception as e:
        # Removed print statements to prevent console output
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ✅ View blockchain data
@app.route('/chain', methods=['GET'])  # 👈 changed from '/view' → '/chain'
def view_data():
    try:
        chain_data = view_blockchain()
        return jsonify({"chain": chain_data}), 200
    except Exception as e:
        # Removed print statements to prevent console output
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run Flask app on 127.0.0.1:5000
    app.run(host='127.0.0.1', port=5000, debug=True)