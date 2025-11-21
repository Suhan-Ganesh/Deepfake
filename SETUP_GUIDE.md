# Deepfake Detection & Blockchain Integration

## Overview
This system analyzes uploaded images and videos for deepfake detection, generates SHA-256 hash values, and stores the results on the blockchain.

## Features
- ✅ Deepfake detection for images and videos
- ✅ SHA-256 hash generation for all media files
- ✅ Blockchain storage of detection results
- ✅ Support for both authentic and deepfake media
- ✅ Confidence scoring for detection results
- ✅ Ensemble detection using multiple models for improved accuracy
- ✅ Model training capabilities for all supported architectures

## Setup Instructions

### Backend Setup

1. **Install Python Dependencies**
   ```bash
   cd deepfake-chain/backend
   pip install -r requirements.txt
   ```

2. **Configure Blockchain Connection**
   Edit `blockchain_connect.py` and add your:
   - Infura URL
   - Contract Address
   - Account Address
   - Private Key

3. **Add Pre-trained Models (Optional but Recommended)**
   - The system supports multiple deepfake detection models:
     - **MesoNet** - Lightweight CNN for deepfake detection
     - **MesoInception** - Improved version of MesoNet with inception blocks
     - **EfficientNet-ViT** - Hybrid model combining EfficientNet and Vision Transformer
     - **Xception** - Transfer learning based detector
   - Place your pre-trained model files (.h5 for TensorFlow models, .pt/.pth for PyTorch models) in the `models` directory
   - Supported model files:
     - `models/Meso4_DF.h5` - MesoNet trained on DeepFake dataset
     - `models/Meso4_F2F.h5` - MesoNet trained on Face2Face dataset
     - `models/MesoInception_DF.h5` - MesoInception trained on DeepFake dataset
     - `models/MesoInception_F2F.h5` - MesoInception trained on Face2Face dataset
   - The system will automatically detect and load all available models
   - Without models, the system uses a fallback heuristic method

4. **Run Backend**
   ```bash
   python app.py
   ```

### Frontend Setup

1. **Install Dependencies**
   ```bash
   cd deepfake-ui
   npm install
   ```

2. **Run Frontend**
   ```bash
   npm start
   ```

## Using Pre-trained Models

### Recommended Models:
1. **MesoNet** - Lightweight CNN for deepfake detection
2. **XceptionNet** - Transfer learning based detector
3. **EfficientNet-B0** - Efficient and accurate

### Ensemble Detection
The system now uses an ensemble approach that combines predictions from all available models:
- TensorFlow models (MesoNet, MesoInception)
- PyTorch models (EfficientNet-ViT, Xception)
- The final prediction is an average of all model predictions
- This approach provides better accuracy and robustness than using a single model

## Training Your Own Models

The system includes a comprehensive training script ([train_ensemble.py](file:///c:/Users/xxtri/Desktop/Deepfake/deepfake-chain/backend/train_ensemble.py)) that allows you to train all supported models:
- Supports training MesoNet, MesoInception, EfficientNet-ViT, and Xception
- Includes data loading, preprocessing, and augmentation
- Provides performance metrics and model evaluation
- Saves trained models in the correct format for immediate use

See [TRAINING_GUIDE.md](file:///c:/Users/xxtri/Desktop/Deepfake/deepfake-chain/backend/TRAINING_GUIDE.md) for detailed training instructions.

## API Endpoints

### POST /upload
Upload an image or video for deepfake detection and blockchain storage.

**Request:**
- Form data with file field named 'file'

**Response:**
```json
{
  "message": "File successfully analyzed and uploaded to blockchain",
  "transaction_hash": "0x...",
  "file_hash": "SHA256 hash of the file",
  "filename": "original filename",
  "file_type": "image or video",
  "is_deepfake": true/false,
  "confidence": 0.95,
  "detection_method": "ensemble_4_models"
}
```

### GET /chain
View all blockchain records.

**Response:**
```json
{
  "chain": [
    {
      "hash": "SHA256 hash",
      "filename": "filename",
      "file_type": "image or video",
      "is_deepfake": true/false,
      "confidence": 0.95,
      "detection_method": "ensemble_4_models"
    }
  ]
}
```

## Security Notes
- Hash values are generated for ALL media (deepfake or authentic)
- Blockchain provides immutable audit trail
- Detection confidence helps assess reliability
- Multiple detection methods are combined for better accuracy