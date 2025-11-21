# Deepfake Detection Ensemble Training Guide

## Overview
This guide explains how to train the ensemble of deepfake detection models in your system. The training script supports all four models in your ensemble:
- MesoNet
- MesoInception
- EfficientNet-ViT
- Xception

## Prerequisites
1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset in the following structure:
   ```
   dataset/
   ├── real/
   │   ├── real_image1.jpg
   │   ├── real_image2.png
   │   └── ...
   └── fake/
       ├── fake_image1.jpg
       ├── fake_image2.png
       └── ...
   ```

## Training Script Usage

### Basic Training
To train all models with default parameters:
```bash
python train_ensemble.py --data_dir /path/to/your/dataset
```

### Selective Training
To train specific models only:
```bash
python train_ensemble.py --data_dir /path/to/your/dataset --models mesonet meso_inception
```

### Custom Training Parameters
To customize training parameters:
```bash
python train_ensemble.py --data_dir /path/to/your/dataset --epochs 100 --batch_size 16 --learning_rate 0.0001
```

### Full Command Options
```bash
python train_ensemble.py --data_dir DATASET_PATH [--models MODEL1 MODEL2 ...] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
```

Arguments:
- `--data_dir`: Path to your dataset directory (required)
- `--models`: List of models to train (options: mesonet, meso_inception, efficientnet_vit, xception, all)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for optimizers (default: 0.001)

## Model-Specific Details

### TensorFlow Models (MesoNet, MesoInception)
- Image size: 256x256 pixels
- Loss function: Binary crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Callbacks: Model checkpoint, early stopping, learning rate reduction

### PyTorch Models (EfficientNet-ViT, Xception)
- EfficientNet-ViT image size: 224x224 pixels
- Xception image size: 299x299 pixels
- Loss function: BCE with logits
- Optimizer: Adam
- Progress logging every 10 batches

## Output Files
Trained models and related files are saved in the `models/` directory:
- `mesonet_best.h5`: Best MesoNet model (based on validation loss)
- `mesonet_final.h5`: Final MesoNet model after training
- `mesonet_history.json`: Training history for MesoNet
- `meso_inception_best.h5`: Best MesoInception model
- `meso_inception_final.h5`: Final MesoInception model
- `meso_inception_history.json`: Training history for MesoInception
- `efficientnet_vit.pth`: Trained EfficientNet-ViT model
- `xception.pth`: Trained Xception model
- `training_results.json`: Summary of training results for all models

## Using Trained Models
After training, your trained models will be automatically detected and loaded by the existing detection system:
1. Place trained `.h5` files in the `models/` directory for TensorFlow models
2. Place trained `.pth` files in the `models/` directory for PyTorch models
3. Restart your detection service (`python app.py`)
4. The system will automatically load all available models for ensemble detection

## Performance Evaluation
The training script automatically evaluates model performance and saves metrics in the training results file. Key metrics include:
- Accuracy
- Precision
- Recall
- F1 Score

## Troubleshooting
1. **CUDA Out of Memory**: Reduce batch size
2. **Dataset Not Found**: Check dataset directory structure
3. **Model Loading Errors**: Ensure model files are compatible with the architecture
4. **Training Slow**: Consider using GPU acceleration if available

## Tips for Better Training
1. Use a balanced dataset with equal numbers of real and fake samples
2. Ensure images are of good quality and representative of your target domain
3. Start with a lower learning rate (0.0001) for fine-tuning pre-trained models
4. Monitor validation loss to prevent overfitting
5. Use early stopping to automatically stop training when performance plateaus