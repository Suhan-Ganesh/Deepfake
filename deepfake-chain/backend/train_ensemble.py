#!/usr/bin/env python3
"""
Comprehensive training script for the deepfake detection ensemble system.
Supports training of MesoNet, MesoInception, EfficientNet-ViT, and Xception models.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import json
import logging

# Import our models
from mesonet import MesoNet, MesoInception
from advanced_models import EfficientNetViTModel, XceptionModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DeepfakeDataset(Dataset):
    """Custom dataset class for loading deepfake images"""
    
    def __init__(self, data_dir, transform=None, img_size=(224, 224)):
        """
        Args:
            data_dir (str): Path to the dataset directory
            transform (callable, optional): Optional transform to be applied on a sample
            img_size (tuple): Size to resize images to
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Assuming directory structure:
        # data_dir/
        #   real/ -> contains real images
        #   fake/ -> contains deepfake images
        
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')
        
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(real_dir, filename))
                    self.labels.append(0)  # 0 for real
        
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(fake_dir, filename))
                    self.labels.append(1)  # 1 for fake
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.img_size)
        
        # Convert to tensor
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)

def load_dataset_tf(data_dir, img_size=(256, 256), batch_size=32):
    """
    Load dataset for TensorFlow models
    
    Args:
        data_dir (str): Path to dataset directory
        img_size (tuple): Image size for resizing
        batch_size (int): Batch size for training
    
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    # Create image dataset from directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    
    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def load_validation_dataset_tf(data_dir, img_size=(256, 256), batch_size=32):
    """
    Load validation dataset for TensorFlow models
    
    Args:
        data_dir (str): Path to dataset directory
        img_size (tuple): Image size for resizing
        batch_size (int): Batch size for training
    
    Returns:
        tf.data.Dataset: TensorFlow validation dataset
    """
    # Create image dataset from directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    
    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def create_callbacks(model_name):
    """
    Create callbacks for model training
    
    Args:
        model_name (str): Name of the model for checkpoint naming
    
    Returns:
        list: List of Keras callbacks
    """
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Model checkpoint callback
    checkpoint = ModelCheckpoint(
        filepath=f'models/{model_name}_best.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    return [checkpoint, early_stop, reduce_lr]

def train_mesonet(data_dir, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train MesoNet model
    
    Args:
        data_dir (str): Path to dataset directory
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    logger.info("Training MesoNet model...")
    
    # Load datasets
    train_dataset = load_dataset_tf(data_dir, img_size=(256, 256), batch_size=batch_size)
    val_dataset = load_validation_dataset_tf(data_dir, img_size=(256, 256), batch_size=batch_size)
    
    # Create model
    model = MesoNet(input_shape=(256, 256, 3), num_classes=1)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks('mesonet')
    
    # Train model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/mesonet_final.h5')
    
    # Save training history
    with open('models/mesonet_history.json', 'w') as f:
        json.dump(history.history, f)
    
    logger.info("MesoNet training completed!")
    return model, history

def train_meso_inception(data_dir, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train MesoInception model
    
    Args:
        data_dir (str): Path to dataset directory
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    logger.info("Training MesoInception model...")
    
    # Load datasets
    train_dataset = load_dataset_tf(data_dir, img_size=(256, 256), batch_size=batch_size)
    val_dataset = load_validation_dataset_tf(data_dir, img_size=(256, 256), batch_size=batch_size)
    
    # Create model
    model = MesoInception(input_shape=(256, 256, 3), num_classes=1)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks('meso_inception')
    
    # Train model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/meso_inception_final.h5')
    
    # Save training history
    with open('models/meso_inception_history.json', 'w') as f:
        json.dump(history.history, f)
    
    logger.info("MesoInception training completed!")
    return model, history

def train_efficientnet_vit(data_dir, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train EfficientNet-ViT model
    
    Args:
        data_dir (str): Path to dataset directory
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    logger.info("Training EfficientNet-ViT model...")
    
    # Create model
    model_wrapper = EfficientNetViTModel()
    if not model_wrapper.model_loaded:
        logger.error("Failed to initialize EfficientNet-ViT model")
        return None, None
    
    model = model_wrapper.model
    device = model_wrapper.device
    
    # Move model to device
    model.to(device)
    
    # Create dataset
    dataset = DeepfakeDataset(data_dir, img_size=(224, 224))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device).float()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output.squeeze(), target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        logger.info(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'models/efficientnet_vit.pth')
    
    logger.info("EfficientNet-ViT training completed!")
    return model, train_losses

def train_xception(data_dir, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train Xception model
    
    Args:
        data_dir (str): Path to dataset directory
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    logger.info("Training Xception model...")
    
    # Create model
    model_wrapper = XceptionModel()
    if not model_wrapper.model_loaded:
        logger.error("Failed to initialize Xception model")
        return None, None
    
    model = model_wrapper.model
    device = model_wrapper.device
    
    # Move model to device
    model.to(device)
    
    # Create dataset
    dataset = DeepfakeDataset(data_dir, img_size=(299, 299))  # Xception expects 299x299
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device).float()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output.squeeze(), target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        logger.info(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'models/xception.pth')
    
    logger.info("Xception training completed!")
    return model, train_losses

def evaluate_model_tf(model, test_dataset):
    """
    Evaluate TensorFlow model
    
    Args:
        model: Trained TensorFlow model
        test_dataset: Test dataset
    
    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    predictions = model.predict(test_dataset)
    predicted_labels = (predictions > 0.5).astype(int)
    
    # True labels
    true_labels = []
    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    return metrics

def evaluate_model_torch(model, dataloader, device):
    """
    Evaluate PyTorch model
    
    Args:
        model: Trained PyTorch model
        dataloader: Test data loader
        device: Device to run evaluation on
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = torch.sigmoid(output).squeeze()
            predicted_labels = (predictions > 0.5).int()
            
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    return metrics

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train deepfake detection models')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--models', nargs='+', default=['all'], 
                        choices=['all', 'mesonet', 'meso_inception', 'efficientnet_vit', 'xception'],
                        help='Models to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory {args.data_dir} does not exist")
        return
    
    # Models to train
    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['mesonet', 'meso_inception', 'efficientnet_vit', 'xception']
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Dictionary to store training results
    results = {}
    
    # Train MesoNet
    if 'mesonet' in models_to_train:
        try:
            model, history = train_mesonet(
                args.data_dir, 
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate
            )
            results['mesonet'] = {
                'status': 'success',
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'val_loss': history.history['val_loss'][-1],
                'val_accuracy': history.history['val_accuracy'][-1]
            }
        except Exception as e:
            logger.error(f"Error training MesoNet: {str(e)}")
            results['mesonet'] = {'status': 'failed', 'error': str(e)}
    
    # Train MesoInception
    if 'meso_inception' in models_to_train:
        try:
            model, history = train_meso_inception(
                args.data_dir, 
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate
            )
            results['meso_inception'] = {
                'status': 'success',
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'val_loss': history.history['val_loss'][-1],
                'val_accuracy': history.history['val_accuracy'][-1]
            }
        except Exception as e:
            logger.error(f"Error training MesoInception: {str(e)}")
            results['meso_inception'] = {'status': 'failed', 'error': str(e)}
    
    # Train EfficientNet-ViT
    if 'efficientnet_vit' in models_to_train:
        try:
            model, losses = train_efficientnet_vit(
                args.data_dir, 
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate
            )
            if model is not None:
                results['efficientnet_vit'] = {
                    'status': 'success',
                    'final_loss': losses[-1] if losses else 0.0
                }
            else:
                results['efficientnet_vit'] = {'status': 'failed', 'error': 'Model initialization failed'}
        except Exception as e:
            logger.error(f"Error training EfficientNet-ViT: {str(e)}")
            results['efficientnet_vit'] = {'status': 'failed', 'error': str(e)}
    
    # Train Xception
    if 'xception' in models_to_train:
        try:
            model, losses = train_xception(
                args.data_dir, 
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate
            )
            if model is not None:
                results['xception'] = {
                    'status': 'success',
                    'final_loss': losses[-1] if losses else 0.0
                }
            else:
                results['xception'] = {'status': 'failed', 'error': 'Model initialization failed'}
        except Exception as e:
            logger.error(f"Error training Xception: {str(e)}")
            results['xception'] = {'status': 'failed', 'error': str(e)}
    
    # Save results
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed!")
    logger.info("Results:")
    for model_name, result in results.items():
        logger.info(f"  {model_name}: {result['status']}")

if __name__ == "__main__":
    main()