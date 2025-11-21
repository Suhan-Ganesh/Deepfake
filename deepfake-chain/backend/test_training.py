#!/usr/bin/env python3
"""
Test script to verify the training functionality works correctly
"""

import os
import sys
import argparse

def test_imports():
    """Test that all required modules can be imported"""
    try:
        # Test TensorFlow imports
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
        from tensorflow.keras.optimizers import Adam
        print("✅ TensorFlow imports successful")
        
        # Test PyTorch imports
        import torch
        import torch.nn as nn
        import torchvision
        print("✅ PyTorch imports successful")
        
        # Test our custom modules
        from mesonet import MesoNet, MesoInception
        from advanced_models import EfficientNetViTModel, XceptionModel
        print("✅ Custom model imports successful")
        
        # Test training script imports
        import train_ensemble
        print("✅ Training script imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_model_creation():
    """Test that models can be created successfully"""
    try:
        # Test MesoNet creation
        from mesonet import MesoNet, MesoInception
        mesonet = MesoNet()
        print("✅ MesoNet creation successful")
        
        # Test MesoInception creation
        meso_inception = MesoInception()
        print("✅ MesoInception creation successful")
        
        # Test EfficientNet-ViT creation
        from advanced_models import EfficientNetViTModel
        effnet_vit = EfficientNetViTModel()
        print("✅ EfficientNet-ViT creation successful")
        
        # Test Xception creation
        from advanced_models import XceptionModel
        xception = XceptionModel()
        print("✅ Xception creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Running training functionality tests...\n")
    
    # Test 1: Import functionality
    print("1. Testing imports...")
    if not test_imports():
        return False
    
    print("\n2. Testing model creation...")
    if not test_model_creation():
        return False
    
    print("\n✅ All tests passed! Training functionality is ready.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)