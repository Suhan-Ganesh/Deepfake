import cv2
import numpy as np
from typing import Optional, Tuple, Any
import os
from PIL import Image
import io
import time

# Try to import TensorFlow components
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model  # type: ignore
    from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
    TENSORFLOW_AVAILABLE = True

    # Try to import MesoNet
    try:
        from mesonet import MesoNet, MesoInception, load_pretrained_mesonet
        MESONET_AVAILABLE = True
    except ImportError:
        MESONET_AVAILABLE = False
except ImportError:
    TENSORFLOW_AVAILABLE = False
    load_model = None  # type: ignore
    img_to_array = None  # type: ignore
    tf = None  # type: ignore
    MESONET_AVAILABLE = False

# Try to import PyTorch components for advanced models
try:
    import torch
    import timm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import Hugging Face components for enhanced detection
try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Try to import the enhanced deepfake detector
try:
    from enhanced_deepfake_detector import EnhancedDeepfakeDetector
    ENHANCED_DETECTOR_AVAILABLE = True
except ImportError:
    ENHANCED_DETECTOR_AVAILABLE = False


class DeepfakeDetector:
    def __init__(self, model_path=None, force_fallback=False, model_paths=None, use_enhanced_detection=False):
        """
        Initialize the deepfake detector.
        If no model is provided, it will use a simple CNN-based detection.
        """
        self.model = None
        self.models = []  # Support for multiple TensorFlow models
        self.advanced_models = []  # Support for PyTorch models (EfficientNet-ViT, Xception)
        self.model_names = []  # Track model names for ensemble
        self.model_loaded = False
        self.force_fallback = force_fallback  # New parameter to force fallback mode
        self.use_enhanced_detection = use_enhanced_detection  # Use enhanced Hugging Face models
        self.hf_client = None
        self.enhanced_detector = None
        
        # Initialize Hugging Face client if enhanced detection is enabled
        if self.use_enhanced_detection and HUGGINGFACE_AVAILABLE:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                try:
                    self.hf_client = InferenceClient(
                        provider="hf-inference",
                        api_key=hf_token
                    )
                except Exception:
                    self.hf_client = None
        
        # Initialize enhanced detector if available
        if ENHANCED_DETECTOR_AVAILABLE:
            try:
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    self.enhanced_detector = EnhancedDeepfakeDetector(hf_token=hf_token)
            except Exception:
                self.enhanced_detector = None

        # Detection parameters
        self.detection_threshold = 0.5  # Threshold for classifying as deepfake

        # Performance tracking for best models
        self.image_model_performance = {}  # Track model performance for images
        self.video_model_performance = {}  # Track model performance for videos

        if not TENSORFLOW_AVAILABLE and not TORCH_AVAILABLE and not self.hf_client and not self.enhanced_detector:
            return

        if force_fallback:
            return

        self._load_all_models(model_path, model_paths)

    def _load_all_models(self, model_path=None, model_paths=None):
        """Load all available models (TensorFlow and PyTorch)"""
        self._load_tensorflow_models(model_path, model_paths)
        self._load_pytorch_models()

        total_models = len(self.models) + len(self.advanced_models)
        if total_models > 0:
            self.model_loaded = True

    def _load_tensorflow_models(self, model_path=None, model_paths=None):
        """Load TensorFlow-based models (MesoNet, MesoInception)"""
        if not TENSORFLOW_AVAILABLE:
            return

        if model_paths is None and model_path is None:
            model_files = []
            if os.path.exists("models") and os.path.isdir("models"):
                for f in os.listdir("models"):
                    if f.endswith('.h5'):
                        model_files.append(os.path.join("models", f))
            for f in os.listdir("."):
                if f.endswith('.h5') and f.startswith('mesonet'):
                    model_files.append(f)
            if model_files:
                filtered_model_files = [f for f in model_files if 'MesoInception' not in f]
                if filtered_model_files:
                    model_paths = filtered_model_files
                else:
                    model_paths = model_files

        if model_paths:
            self._load_multiple_tensorflow_models(model_paths)
        elif model_path:
            self._load_single_tensorflow_model(model_path)
        else:
            self._create_default_mesonet_models()

    def _create_default_mesonet_models(self):
        if MESONET_AVAILABLE and 'MesoNet' in globals():
            try:
                mesonet = MesoNet()
                self.models.append(mesonet)
                self.model_names.append("MesoNet")

                mesoinception = MesoInception()
                self.models.append(mesoinception)
                self.model_names.append("MesoInception")
            except Exception:
                pass

    def _load_single_tensorflow_model(self, model_path):
        if MESONET_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                if MESONET_AVAILABLE and 'MesoNet' in globals() and 'MesoInception' in globals():
                    if 'Inception' in model_path:
                        model = MesoInception()
                        model_name = "MesoInception"
                    else:
                        model = MesoNet()
                        model_name = "MesoNet"
                    try:
                        model.load_weights(model_path)
                        self.models.append(model)
                        self.model_names.append(model_name)
                        return
                    except Exception:
                        model = None
                if MESONET_AVAILABLE and 'load_pretrained_mesonet' in globals():
                    model_func = globals().get('load_pretrained_mesonet')
                    if model_func:
                        model = model_func(model_path)
                        if model is not None:
                            self.models.append(model)
                            self.model_names.append("MesoNet_Loaded")
                            return
                if model_path and os.path.exists(model_path):
                    custom_objects = {}
                    if TENSORFLOW_AVAILABLE and tf is not None:
                        try:
                            if hasattr(tf, 'keras') and hasattr(getattr(tf, 'keras'), 'optimizers') and hasattr(getattr(getattr(tf, 'keras'), 'optimizers'), 'schedules'):
                                schedules = getattr(getattr(tf, 'keras'), 'optimizers').schedules
                                if hasattr(schedules, 'ExponentialDecay'):
                                    custom_objects['ExponentialDecay'] = schedules.ExponentialDecay
                        except:
                            pass
                    if load_model is not None:
                        model = load_model(model_path, custom_objects=custom_objects)
                        if model is not None:
                            self.models.append(model)
                            self.model_names.append("TensorFlow_Model")
            except Exception:
                pass

    def _load_multiple_tensorflow_models(self, model_paths):
        loaded_models = []
        loaded_model_names = []
        for path in model_paths:
            if os.path.exists(path):
                try:
                    if MESONET_AVAILABLE and 'MesoNet' in globals() and 'MesoInception' in globals():
                        if 'Inception' in path:
                            model = MesoInception()
                            model_name = "MesoInception"
                        else:
                            model = MesoNet()
                            model_name = "MesoNet"
                        try:
                            model.load_weights(path)
                            loaded_models.append(model)
                            loaded_model_names.append(model_name)
                            continue
                        except Exception:
                            pass
                    if MESONET_AVAILABLE and 'load_pretrained_mesonet' in globals():
                        model_func = globals().get('load_pretrained_mesonet')
                        if model_func:
                            model = model_func(path)
                            if model is not None:
                                loaded_models.append(model)
                                loaded_model_names.append("MesoNet_Loaded")
                                continue
                    custom_objects = {}
                    if TENSORFLOW_AVAILABLE and tf is not None:
                        try:
                            if hasattr(tf, 'keras') and hasattr(getattr(tf, 'keras'), 'optimizers') and hasattr(getattr(getattr(tf, 'keras'), 'optimizers'), 'schedules'):
                                schedules = getattr(getattr(tf, 'keras'), 'optimizers').schedules
                                if hasattr(schedules, 'ExponentialDecay'):
                                    custom_objects['ExponentialDecay'] = schedules.ExponentialDecay
                        except:
                            pass
                    if load_model is not None:
                        model = load_model(path, custom_objects=custom_objects)
                        if model is not None:
                            loaded_models.append(model)
                            loaded_model_names.append("TensorFlow_Model")
                except Exception:
                    pass
        if loaded_models:
            self.models.extend(loaded_models)
            self.model_names.extend(loaded_model_names)

    def _load_pytorch_models(self):
        if not TORCH_AVAILABLE:
            return
        try:
            from advanced_models import EfficientNetViTModel, XceptionModel
            try:
                effnet_vit = EfficientNetViTModel()
                if effnet_vit.model_loaded:
                    self.advanced_models.append(effnet_vit)
                    self.model_names.append("EfficientNet-ViT")
            except Exception:
                pass
            try:
                xception = XceptionModel()
                if xception.model_loaded:
                    self.advanced_models.append(xception)
                    self.model_names.append("Xception")
            except Exception:
                pass
        except ImportError:
            pass
        except Exception:
            pass

    def preprocess_image(self, image_data):
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((256, 256))
            if TENSORFLOW_AVAILABLE:
                img_array = img_to_array(image)
            else:
                img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception:
            return None

    def preprocess_image_for_torch(self, image_data, size=(224, 224)):
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(size)
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0)
            return image_tensor
        except Exception:
            return None

    def preprocess_video(self, video_data):
        try:
            temp_path = "temp_video.mp4"
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            cap = cv2.VideoCapture(temp_path)
            frames = []
            frame_count = 0
            max_frames = 10
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (256, 256))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                frame_count += 1
            cap.release()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return np.array(frames) if frames else None
        except Exception:
            return None

    def _run_hf_inference(self, image: Image.Image, model_name: str) -> list:
        """
        Run inference on Hugging Face models
        
        Args:
            image (PIL.Image): Preprocessed image
            model_name (str): Name of the model to use
            
        Returns:
            list: Model predictions
        """
        if not self.hf_client:
            return []
            
        try:
            if model_name == "ai_image_detector":
                model_id = "umm-maybe/AI-image-detector"
            elif model_name == "efficientnet_b7":
                model_id = "google/efficientnet-b7"
            else:
                return []
                
            result = self.hf_client.image_classification(image, model=model_id)
            return result
        except Exception as e:
            return []

    def _extract_deepfake_confidence(self, predictions: list, model_name: str) -> float:
        """
        Extract deepfake confidence from model predictions
        
        Args:
            predictions (list): Model predictions
            model_name (str): Name of the model
            
        Returns:
            float: Confidence score for deepfake detection
        """
        if not predictions:
            return 0.5  # Neutral confidence if no predictions
            
        try:
            if model_name == "ai_image_detector":
                # For AI-image-detector, look for "artificial" or similar labels
                for pred in predictions:
                    if "artificial" in pred["label"].lower() or "fake" in pred["label"].lower():
                        return pred["score"]
                    elif "real" in pred["label"].lower():
                        return 1.0 - pred["score"]  # Invert for deepfake confidence
            
            elif model_name == "efficientnet_b7":
                # For EfficientNet-B7, we might need to interpret the labels
                for pred in predictions:
                    if "fake" in pred["label"].lower() or "deepfake" in pred["label"].lower():
                        return pred["score"]
                return predictions[0]["score"] if predictions else 0.5
                
        except Exception as e:
            pass
            
        return 0.5  # Default neutral confidence

    def _detect_with_enhanced_models(self, image_data):
        """
        Detect deepfakes using enhanced Hugging Face models
        
        Args:
            image_data (bytes): Image data
            
        Returns:
            tuple: (is_deepfake, confidence, method)
        """
        # Prefer the new enhanced detector if available
        if self.enhanced_detector:
            try:
                # Save image data to a temporary file
                temp_path = "temp_enhanced_analysis.jpg"
                with open(temp_path, "wb") as f:
                    f.write(image_data)
                
                # Run analysis with enhanced detector
                result = self.enhanced_detector.analyze_image(temp_path)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if "error" not in result:
                    confidence = result["final_prediction"]
                    is_deepfake = result["is_deepfake"]
                    method = f"enhanced_detector (raw: {result['raw_ensemble_score']:.4f}, calibrated: {confidence:.4f})"
                    return is_deepfake, confidence, method
            except Exception as e:
                pass  # Fall back to HF client approach
        
        # Fallback to direct HF client approach
        if not self.hf_client:
            return False, 0.0, "enhanced_detection_unavailable"
            
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Run inference on both models
            ai_detector_result = self._run_hf_inference(image, "ai_image_detector")
            effnet_result = self._run_hf_inference(image, "efficientnet_b7")
            
            # Extract confidences
            ai_confidence = self._extract_deepfake_confidence(ai_detector_result, "ai_image_detector")
            effnet_confidence = self._extract_deepfake_confidence(effnet_result, "efficientnet_b7")
            
            # Average the confidences
            confidence = (ai_confidence + effnet_confidence) / 2.0
            is_deepfake = confidence > self.detection_threshold
            
            method = f"enhanced_hf_models (AI-detector: {ai_confidence:.4f}, EfficientNet-B7: {effnet_confidence:.4f})"
            
            return is_deepfake, confidence, method
            
        except Exception as e:
            return False, 0.0, f"enhanced_detection_error: {str(e)}"

    def detect_image(self, image_data):
        # If enhanced detection is enabled and available, use it
        if self.use_enhanced_detection and (self.enhanced_detector or self.hf_client):
            return self._detect_with_enhanced_models(image_data)
        
        try:
            img_array = self.preprocess_image(image_data)
            if img_array is None:
                return False, 0.0, "preprocessing_failed"
            if self.model_loaded and not self.force_fallback:
                predictions = []
                model_details = []
                model_confidences = []  # Track individual model confidences
                
                # Process TensorFlow models
                for i, model in enumerate(self.models):
                    try:
                        pred = model.predict(img_array, verbose=0)
                        pred_value = float(pred[0][0])
                        if 0 <= pred_value <= 1:
                            predictions.append(pred_value)
                            model_confidences.append((self.model_names[i], pred_value))
                            model_details.append(f"{self.model_names[i]}: {pred_value:.4f}")
                    except Exception:
                        pass
                
                # Process PyTorch models
                for i, model in enumerate(self.advanced_models):
                    try:
                        is_deepfake, confidence = model.detect(image_data)
                        predictions.append(confidence)
                        model_confidences.append((self.model_names[len(self.models) + i], confidence))
                        model_details.append(f"{self.model_names[len(self.models) + i]}: {confidence:.4f}")
                    except Exception:
                        pass
                
                if predictions:
                    confidence = float(np.mean(predictions))
                    is_deepfake = confidence > self.detection_threshold
                    
                    # Track best performing model for images
                    if model_confidences:
                        best_model = max(model_confidences, key=lambda x: x[1])
                        best_model_name, best_model_confidence = best_model
                        
                        # Update performance tracking
                        if best_model_name not in self.image_model_performance:
                            self.image_model_performance[best_model_name] = []
                        self.image_model_performance[best_model_name].append(best_model_confidence)
                        
                        # Find overall best model for images
                        best_image_model = self._get_best_model_for_images()
                        method = f"ensemble_{len(predictions)}_models"
                        if best_image_model:
                            method += f" (best: {best_image_model})"
                    else:
                        best_image_model = None
                        method = f"ensemble_{len(predictions)}_models"
                    
                    if model_details:
                        method += f" ({', '.join(model_details)})"
                        method += f" (average: {confidence:.4f})"
                    return is_deepfake, confidence, method
                else:
                    return self._fallback_image_detection(img_array)
            else:
                return self._fallback_image_detection(img_array)
        except Exception:
            return False, 0.0, "error"

    def detect_video(self, video_data):
        try:
            frames = self.preprocess_video(video_data)
            if frames is None or len(frames) == 0:
                return False, 0.0, "preprocessing_failed"
            if self.model_loaded and not self.force_fallback:
                frame_predictions = []
                frame_model_predictions = []  # Track predictions per frame per model
                
                for frame in frames:
                    try:
                        frame_array = np.expand_dims(frame, axis=0)
                        predictions = []
                        frame_model_data = []  # Track model predictions for this frame
                        
                        # Process TensorFlow models
                        for i, model in enumerate(self.models):
                            try:
                                pred = model.predict(frame_array, verbose=0)
                                pred_value = float(pred[0][0])
                                if 0 <= pred_value <= 1:
                                    predictions.append(pred_value)
                                    frame_model_data.append((self.model_names[i], pred_value))
                            except Exception:
                                pass
                        
                        if predictions:
                            frame_avg = float(np.mean(predictions))
                            frame_predictions.append(frame_avg)
                            frame_model_predictions.append(frame_model_data)
                    except Exception:
                        continue
                
                if frame_predictions:
                    confidence = float(np.mean(frame_predictions))
                    is_deepfake = confidence > self.detection_threshold
                    
                    # Track best performing model for videos
                    if frame_model_predictions:
                        # Find best model across all frames
                        model_performance = {}
                        for frame_models in frame_model_predictions:
                            if frame_models:
                                best_in_frame = max(frame_models, key=lambda x: x[1])
                                model_name, model_conf = best_in_frame
                                if model_name not in model_performance:
                                    model_performance[model_name] = []
                                model_performance[model_name].append(model_conf)
                        
                        # Update video performance tracking
                        for model_name, confidences in model_performance.items():
                            if model_name not in self.video_model_performance:
                                self.video_model_performance[model_name] = []
                            self.video_model_performance[model_name].extend(confidences)
                        
                        # Find overall best model for videos
                        best_video_model = self._get_best_model_for_videos()
                        method = f"video_{len(frame_predictions)}_frames"
                        if best_video_model:
                            method += f" (best: {best_video_model})"
                    else:
                        best_video_model = None
                        method = f"video_{len(frame_predictions)}_frames"
                    
                    method += f" (average: {confidence:.4f})"
                    return is_deepfake, confidence, method
                else:
                    return self._fallback_video_detection(frames)
            else:
                return self._fallback_video_detection(frames)
        except Exception:
            return False, 0.0, "error"

    def detect(self, file_data, file_type):
        if file_type == 'video':
            return self.detect_video(file_data)
        else:
            return self.detect_image(file_data)

    def _fallback_image_detection(self, img_array):
        try:
            if img_array is not None:
                mean_val = np.mean(img_array)
                std_val = np.std(img_array)
                if len(img_array.shape) == 4:
                    r_channel = img_array[0, :, :, 0]
                    g_channel = img_array[0, :, :, 1]
                    b_channel = img_array[0, :, :, 2]
                else:
                    r_channel = img_array[:, :, 0]
                    g_channel = img_array[:, :, 1]
                    b_channel = img_array[:, :, 2]
                rg_diff = np.mean(np.abs(r_channel - g_channel))
                rb_diff = np.mean(np.abs(r_channel - b_channel))
                gb_diff = np.mean(np.abs(g_channel - b_channel))
                color_balance = (rg_diff + rb_diff + gb_diff) / 3
                texture_smoothness = std_val
                if texture_smoothness < 0.15 and color_balance < 0.2:
                    confidence = 0.8
                    return True, confidence, "fallback_heuristic_improved"
                elif texture_smoothness < 0.2 and color_balance < 0.3:
                    confidence = 0.6
                    return True, confidence, "fallback_heuristic_improved"
                elif 0.4 < mean_val < 0.6 and std_val < 0.2:
                    confidence = 0.5
                    return True, confidence, "fallback_heuristic_original"
                else:
                    confidence = 0.2
                    return False, confidence, "fallback_heuristic_improved"
            else:
                return False, 0.0, "fallback_no_data"
        except Exception:
            return False, 0.0, "fallback_error"

    def _fallback_video_detection(self, frames):
        try:
            if frames is not None and len(frames) > 0:
                frame_means = [np.mean(frame) for frame in frames]
                frame_stds = [np.std(frame) for frame in frames]
                mean_variation = np.std(frame_means)
                std_variation = np.std(frame_stds)
                if mean_variation > 0.1 or std_variation > 0.05:
                    confidence = 0.7
                    return True, confidence, "fallback_video_improved"
                else:
                    confidence = 0.3
                    return False, confidence, "fallback_video_improved"
            else:
                return False, 0.5, "fallback_video"
        except Exception:
            return False, 0.5, "fallback_video_error"

    def set_detection_parameters(self, threshold=0.5):
        self.detection_threshold = max(0.0, min(1.0, threshold))

    def _get_best_model_for_images(self):
        """Determine the best performing model for images based on average confidence"""
        if not self.image_model_performance:
            return None
        
        model_averages = {}
        for model_name, confidences in self.image_model_performance.items():
            if confidences:
                model_averages[model_name] = np.mean(confidences)
        
        if not model_averages:
            return None
            
        best_model = max(model_averages, key=model_averages.get)
        return best_model

    def _get_best_model_for_videos(self):
        """Determine the best performing model for videos based on average confidence"""
        if not self.video_model_performance:
            return None
        
        model_averages = {}
        for model_name, confidences in self.video_model_performance.items():
            if confidences:
                model_averages[model_name] = np.mean(confidences)
        
        if not model_averages:
            return None
            
        best_model = max(model_averages, key=model_averages.get)
        return best_model

    def get_model_performance_stats(self):
        """Get performance statistics for all models"""
        stats = {
            'image_models': {},
            'video_models': {}
        }
        
        # Image model statistics
        for model_name, confidences in self.image_model_performance.items():
            if confidences:
                stats['image_models'][model_name] = {
                    'count': len(confidences),
                    'average_confidence': float(np.mean(confidences)),
                    'min_confidence': float(np.min(confidences)),
                    'max_confidence': float(np.max(confidences))
                }
        
        # Video model statistics
        for model_name, confidences in self.video_model_performance.items():
            if confidences:
                stats['video_models'][model_name] = {
                    'count': len(confidences),
                    'average_confidence': float(np.mean(confidences)),
                    'min_confidence': float(np.min(confidences)),
                    'max_confidence': float(np.max(confidences))
                }
        
        return stats

    def reset_model_performance_stats(self):
        """Reset all model performance statistics"""
        self.image_model_performance = {}
        self.video_model_performance = {}

    def get_best_models(self):
        """Get the best performing models for images and videos"""
        return {
            'best_image_model': self._get_best_model_for_images(),
            'best_video_model': self._get_best_model_for_videos()
        }


def get_detector(force_fallback=False, model_paths=None):
    return DeepfakeDetector(force_fallback=force_fallback, model_paths=model_paths)
