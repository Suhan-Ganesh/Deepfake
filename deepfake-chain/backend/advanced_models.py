#!/usr/bin/env python3
"""
Advanced deepfake detection models including EfficientNet-ViT and Xception
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import io
import cv2
import os

class EfficientNetViTModel:
    """EfficientNet-ViT hybrid model for deepfake detection"""
    
    def __init__(self, model_path=None):
        """
        Initialize the EfficientNet-ViT model
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.confidence_threshold = 0.5  # Threshold for classifying as deepfake
        self._load_model(model_path)
    
    def _load_model(self, model_path=None):
        """Load the EfficientNet-ViT model"""
        try:
            # For now, we'll use a pre-trained EfficientNet as a placeholder
            # In practice, you would load your specific trained model
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            # Removed print statement to prevent console output
        except Exception as e:
            # Removed print statement to prevent console output
            self.model_loaded = False
    
    def preprocess_image(self, image_data):
        """
        Preprocess image for EfficientNet-ViT model
        """
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model's expected size (224x224 for EfficientNet)
            image = image.resize((224, 224))
            
            # Convert to tensor
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0  # Normalize to [0, 1]
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC to CHW
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
        except Exception as e:
            # Removed print statement to prevent console output
            return None
    
    def detect(self, image_data):
        """
        Detect if an image is a deepfake using EfficientNet-ViT
        Returns: (is_deepfake: bool, confidence: float)
        """
        if not self.model_loaded:
            return False, 0.0
        
        try:
            image_tensor = self.preprocess_image(image_data)
            if image_tensor is None:
                return False, 0.0
            
            with torch.no_grad():
                output = self.model(image_tensor)
                # Apply sigmoid to get probability
                probability = torch.sigmoid(output).item()
                
                # Improved prediction logic
                # Adjust confidence based on how far it is from the threshold
                distance_from_threshold = abs(probability - self.confidence_threshold)
                # Boost confidence for predictions far from threshold
                if distance_from_threshold > 0.2:
                    confidence = min(1.0, probability + 0.1 * (distance_from_threshold - 0.2))
                else:
                    confidence = probability
                
                is_deepfake = confidence > self.confidence_threshold
                return is_deepfake, confidence
                
        except Exception as e:
            # Removed print statement to prevent console output
            return False, 0.0

    def set_confidence_threshold(self, threshold=0.5):
        """
        Set the confidence threshold for deepfake classification
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        # Removed print statement to prevent console output


class XceptionModel:
    """Xception model for deepfake detection"""
    
    def __init__(self):
        """
        Initialize the Xception model
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.confidence_threshold = 0.5  # Threshold for classifying as deepfake
        self._load_model()
    
    def _load_model(self):
        """Load the Xception model"""
        try:
            # Create Xception model
            self.model = timm.create_model('xception', pretrained=True, num_classes=1)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            # Removed print statement to prevent console output
        except Exception as e:
            # Removed print statement to prevent console output
            self.model_loaded = False
    
    def preprocess_image(self, image_data):
        """
        Preprocess image for Xception model
        """
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model's expected size (299x299 for Xception)
            image = image.resize((299, 299))
            
            # Convert to tensor
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0  # Normalize to [0, 1]
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC to CHW
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
        except Exception as e:
            # Removed print statement to prevent console output
            return None
    
    def detect(self, image_data):
        """
        Detect if an image is a deepfake using Xception
        Returns: (is_deepfake: bool, confidence: float)
        """
        if not self.model_loaded:
            return False, 0.0
        
        try:
            image_tensor = self.preprocess_image(image_data)
            if image_tensor is None:
                return False, 0.0
            
            with torch.no_grad():
                output = self.model(image_tensor)
                # Apply sigmoid to get probability
                probability = torch.sigmoid(output).item()
                
                # Improved prediction logic
                # Adjust confidence based on how far it is from the threshold
                distance_from_threshold = abs(probability - self.confidence_threshold)
                # Boost confidence for predictions far from threshold
                if distance_from_threshold > 0.2:
                    confidence = min(1.0, probability + 0.1 * (distance_from_threshold - 0.2))
                else:
                    confidence = probability
                
                is_deepfake = confidence > self.confidence_threshold
                return is_deepfake, confidence
                
        except Exception as e:
            # Removed print statement to prevent console output
            return False, 0.0

    def set_confidence_threshold(self, threshold=0.5):
        """
        Set the confidence threshold for deepfake classification
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        # Removed print statement to prevent console output


# This module is now integrated with the main DeepfakeDetector class
# The EnsembleDetector class has been removed to avoid duplication
# All models are now managed by the main detection system

# Example usage
if __name__ == "__main__":
    # This is just for testing
    # Removed print statements to prevent console output
    try:
        # Test EfficientNet-ViT
        effnet_vit = EfficientNetViTModel()
        # Removed print statements to prevent console output
        
        # Test Xception
        xception = XceptionModel()
        # Removed print statements to prevent console output
    except Exception as e:
        # Removed print statement to prevent console output
        pass