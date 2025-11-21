# Deepfake Detection & Blockchain Integration

This project detects deepfakes in images and videos using an ensemble of AI models, then stores the results on the blockchain for verification.

## Folder Structure

- `backend/`
  Contains all backend source code:
  - `app.py` — Flask application entry point
  - `deepfake_detector.py` — Deepfake detection logic with ensemble support
  - `mesonet.py` — MesoNet and MesoInception model implementations
  - `advanced_models.py` — EfficientNet-ViT and Xception model implementations
  - `blockchain_connect.py` — Blockchain integration logic
  - `train_ensemble.py` — Training script for all models
  - `requirements.txt` — Python dependencies
  - `models/` — Pre-trained model files

## Getting Started

1. **Install dependencies:**
   ```
   pip install -r backend/requirements.txt
   ```

2. **Run the Flask server:**
   ```
   python backend/app.py
   ```

3. **Access the API:**
   The API will be available at [http://localhost:5000](http://localhost:5000)

## Training Models

To train the ensemble of models, use the training script:
```
python backend/train_ensemble.py --data_dir /path/to/dataset
```

See `backend/TRAINING_GUIDE.md` for detailed training instructions.

## Notes

- Python 3.7+ recommended.
- Supports both TensorFlow and PyTorch models.
- All cache and environment files are ignored via `.gitignore`.

##