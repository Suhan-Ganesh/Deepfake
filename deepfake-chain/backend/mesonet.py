"""
MesoNet implementation for DeepFake detection
Based on the paper: "MesoNet: a Compact Facial Video Forgery Detection Network"
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, 
    Flatten, Dense, Dropout, Activation, concatenate
)
from tensorflow.keras.optimizers import Adam

def MesoNet(input_shape=(256, 256, 3), num_classes=1):
    """
    Create the MesoNet model architecture (matching official implementation)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes (1 for binary classification)
        
    Returns:
        MesoNet model
    """
    # Input layer
    inputs = Input(shape=input_shape, name='input_2')
    
    # First convolutional block
    x = Conv2D(8, (3, 3), padding='same', activation='relu', name='conv2d_5')(inputs)
    x = BatchNormalization(name='batch_normalization_5')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_5')(x)
    
    # Second convolutional block
    x = Conv2D(8, (5, 5), padding='same', activation='relu', name='conv2d_6')(x)
    x = BatchNormalization(name='batch_normalization_6')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_6')(x)
    
    # Third convolutional block
    x = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv2d_7')(x)
    x = BatchNormalization(name='batch_normalization_7')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_7')(x)
    
    # Fourth convolutional block
    x = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv2d_8')(x)
    x = BatchNormalization(name='batch_normalization_8')(x)
    x = MaxPooling2D(pool_size=(4, 4), name='max_pooling2d_8')(x)
    
    # Fully connected layers
    x = Flatten(name='flatten_2')(x)
    x = Dropout(0.5, name='dropout_3')(x)
    x = Dense(16, name='dense_3')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5, name='dropout_4')(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='sigmoid', name='dense_4')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def MesoInception(input_shape=(256, 256, 3), num_classes=1):
    """
    Create the MesoInception model architecture (improved version of MesoNet)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes (1 for binary classification)
        
    Returns:
        MesoInception model
    """
    # Input layer
    inputs = Input(shape=input_shape, name='input_1')
    
    # First convolutional block
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2d_1')(inputs)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(x)
    
    # Inception block 1
    branch1x1 = Conv2D(16, (1, 1), padding='same', activation='relu', name='conv2d_2')(x)
    
    branch3x3 = Conv2D(16, (1, 1), padding='same', activation='relu', name='conv2d_3')(x)
    branch3x3 = Conv2D(16, (3, 3), padding='same', activation='relu', name='conv2d_4')(branch3x3)
    
    branch5x5 = Conv2D(16, (1, 1), padding='same', activation='relu', name='conv2d_9')(x)
    branch5x5 = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv2d_10')(branch5x5)
    
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='max_pooling2d_2')(x)
    branch_pool = Conv2D(16, (1, 1), padding='same', activation='relu', name='conv2d_11')(branch_pool)
    
    x = concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=3, name='concatenate_1')
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3')(x)
    
    # Inception block 2
    branch1x1 = Conv2D(16, (1, 1), padding='same', activation='relu', name='conv2d_12')(x)
    
    branch3x3 = Conv2D(16, (1, 1), padding='same', activation='relu', name='conv2d_13')(x)
    branch3x3 = Conv2D(16, (3, 3), padding='same', activation='relu', name='conv2d_14')(branch3x3)
    
    branch5x5 = Conv2D(16, (1, 1), padding='same', activation='relu', name='conv2d_15')(x)
    branch5x5 = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv2d_16')(branch5x5)
    
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='max_pooling2d_4')(x)
    branch_pool = Conv2D(16, (1, 1), padding='same', activation='relu', name='conv2d_17')(branch_pool)
    
    x = concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=3, name='concatenate_2')
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_5')(x)
    
    # Fully connected layers
    x = Flatten(name='flatten_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(16, name='dense_1')(x)
    x = Activation('relu', name='activation_1')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='sigmoid', name='dense_2')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='MesoInception')
    
    return model

def load_pretrained_mesonet(model_path='mesonet.h5'):
    """
    Load a pre-trained MesoNet model
    
    Args:
        model_path: Path to the pre-trained model file
        
    Returns:
        Loaded model or None if loading failed
    """
    try:
        # Try to load with custom objects to handle serialization issues
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers.schedules import ExponentialDecay
        
        custom_objects = {
            'ExponentialDecay': ExponentialDecay,
            'Adam': Adam
        }
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        # Removed print statement to prevent console output
        return model
    except Exception as e:
        # Removed print statement to prevent console output
        # Try without custom objects
        try:
            model = tf.keras.models.load_model(model_path)
            # Removed print statement to prevent console output
            return model
        except Exception as e2:
            # Removed print statement to prevent console output
            return None

def create_mesonet_with_weights(input_shape=(256, 256, 3), num_classes=1):
    """
    Create MesoNet model and initialize with default weights
    This is a placeholder for when you have pre-trained weights
    """
    model = MesoNet(input_shape, num_classes)
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Example usage:
# model = MesoNet()
# model.summary()