from flask import Flask, request, jsonify
from flask_cors import CORS
from blockchain_connect import upload_to_blockchain, view_blockchain
from local_storage import upload_with_local_fallback, view_with_local_fallback
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
        threshold=0.5
    )

app = Flask(__name__)
CORS(app)

# ✅ Root route (test connection)
@app.route('/')
def home():
    return jsonify({
        "status": "Backend running successfully 🚀",
        "endpoints": {
            "POST /upload": "Upload image/video for deepfake detection",
            "GET /chain": "View blockchain data",
            "GET /total-records": "Get total blockchain records count",
            "GET /model-stats": "Get model performance statistics",
            "GET /best-models": "Get best performing models for images and videos",
            "POST /reset-model-stats": "Reset model performance statistics"
        }
    }), 200


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

        # Upload to blockchain with local storage fallback
        print("📤 Uploading to blockchain with local fallback...")
        result = upload_with_local_fallback(file_data, blockchain_data)
        print(f"✅ Blockchain upload process completed with result: {result}")
        
        # Check if result is a duplicate detection response
        if isinstance(result, dict) and result.get("status") == "duplicate":
            return jsonify({
                "message": "File analysis completed",
                "duplicate": True,
                "duplicate_message": result["message"],
                "transaction_hash": result["transaction_hash"],
                "file_hash": result["file_hash"],
                "filename": file.filename,
                "file_type": file_type,
                "is_deepfake": is_deepfake,
                "confidence": round(confidence, 4),
                "detection_method": detection_method
            }), 200

        return jsonify({
            "message": "File successfully analyzed and uploaded to blockchain",
            "transaction_hash": result,
            "file_hash": file_hash,
            "filename": file.filename,
            "file_type": file_type,
            "is_deepfake": is_deepfake,
            "confidence": round(confidence, 4),
            "detection_method": detection_method
        }), 200

    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


# ✅ View blockchain data
@app.route('/chain', methods=['GET'])  # 👈 changed from '/view' → '/chain'
def view_data():
    try:
        print("🔄 Fetching blockchain data with local fallback...")
        chain_data = view_with_local_fallback()
        print(f"✅ Returning {len(chain_data)} records")
        return jsonify({"chain": chain_data}), 200
    except Exception as e:
        print(f"❌ Error fetching blockchain data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to fetch blockchain data: {str(e)}"}), 500

# ✅ Get model performance statistics
@app.route('/model-stats', methods=['GET'])
def get_model_stats():
    try:
        stats = detector.get_model_performance_stats()
        return jsonify({"model_stats": stats}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Get best performing models
@app.route('/best-models', methods=['GET'])
def get_best_models():
    try:
        best_models = detector.get_best_models()
        return jsonify({"best_models": best_models}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Reset model performance statistics
@app.route('/reset-model-stats', methods=['POST'])
def reset_model_stats():
    try:
        detector.reset_model_performance_stats()
        return jsonify({"message": "Model performance statistics reset successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Get total blockchain records count
@app.route('/total-records', methods=['GET'])
def get_total_records():
    try:
        from blockchain_connect import get_total_records
        count = get_total_records()
        return jsonify({"total_records": count}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Check if a file hash exists in the blockchain
@app.route('/check-hash/<file_hash>', methods=['GET'])
def check_file_hash(file_hash):
    try:
        print(f"🔍 Checking if hash {file_hash} exists in blockchain...")
        
        # Try blockchain first
        from blockchain_connect import check_duplicate_by_hash
        exists_in_blockchain = check_duplicate_by_hash(file_hash)
        
        if exists_in_blockchain:
            # Get the record details
            from blockchain_connect import view_blockchain
            blockchain_data = view_blockchain()
            matching_record = None
            
            for record in blockchain_data:
                if record.get("fileHash", "").lower() == file_hash.lower():
                    matching_record = record
                    break
            
            if matching_record:
                return jsonify({
                    "found": True,
                    "message": "File found in blockchain",
                    "record": matching_record
                }), 200
            else:
                # This shouldn't happen, but just in case
                return jsonify({
                    "found": True,
                    "message": "File hash exists in blockchain (but record details not found)",
                    "record": None
                }), 200
        else:
            # Check local storage as fallback
            from local_storage import get_local_records
            local_data = get_local_records()
            matching_record = None
            
            for record in local_data:
                record_hash = record.get("hash", record.get("file_hash", ""))
                if record_hash.lower() == file_hash.lower():
                    matching_record = {
                        "fileHash": record_hash,
                        "isDeepfake": record.get("is_deepfake", False),
                        "confidenceScore": record.get("confidence", 0.0),
                        "uploader": record.get("uploader", "0x0000000000000000000000000000000000000000"),
                        "timestamp": record.get("timestamp", 0)
                    }
                    break
            
            if matching_record:
                return jsonify({
                    "found": True,
                    "message": "File found in local storage",
                    "record": matching_record
                }), 200
            else:
                return jsonify({
                    "found": False,
                    "message": "File not found in blockchain or local storage"
                }), 200
                
    except Exception as e:
        print(f"❌ Error checking file hash: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to check file hash: {str(e)}"}), 500


# ✅ Check transaction status
@app.route('/transaction-status/<tx_hash>', methods=['GET'])
def check_transaction_status(tx_hash):
    try:
        from blockchain_connect import check_transaction_status
        status = check_transaction_status(tx_hash)
        if status is None:
            return jsonify({"error": "Could not check transaction status"}), 500
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run Flask app on 127.0.0.1:5000
    app.run(host='127.0.0.1', port=5000, debug=True)