import cv2
import numpy as np
from typing import Optional, Tuple, Any
import os
from PIL import Image
import io

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


class DeepfakeDetector:
    def __init__(self, model_path=None, force_fallback=False, model_paths=None):
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

        # Improved detection parameters
        self.detection_threshold = 0.5  # Threshold for classifying as deepfake
        self.min_model_agreement = 0.6  # Minimum ratio of models that must agree for high confidence
        self.confidence_boost_factor = 1.2  # Factor to boost confidence when models agree

        if not TENSORFLOW_AVAILABLE and not TORCH_AVAILABLE:
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

    def detect_image(self, image_data):
        try:
            img_array = self.preprocess_image(image_data)
            if img_array is None:
                return False, 0.0, "preprocessing_failed"
            if self.model_loaded and not self.force_fallback:
                predictions = []
                model_details = []
                for i, model in enumerate(self.models):
                    try:
                        pred = model.predict(img_array, verbose=0)
                        pred_value = float(pred[0][0])
                        if 0 <= pred_value <= 1:
                            predictions.append(pred_value)
                            model_details.append(f"{self.model_names[i]}: {pred_value:.4f}")
                    except Exception:
                        pass
                for i, model in enumerate(self.advanced_models):
                    try:
                        is_deepfake, confidence = model.detect(image_data)
                        predictions.append(confidence)
                        model_details.append(f"{self.model_names[len(self.models) + i]}: {confidence:.4f}")
                    except Exception:
                        pass
                if predictions:
                    confidence = float(np.mean(predictions))
                    agreement_count = sum(1 for p in predictions if (p > self.detection_threshold) == (confidence > self.detection_threshold))
                    agreement_ratio = agreement_count / len(predictions) if predictions else 0
                    if agreement_ratio >= self.min_model_agreement:
                        confidence = min(1.0, confidence * self.confidence_boost_factor)
                    is_deepfake = confidence > self.detection_threshold
                    method = f"ensemble_{len(predictions)}_models"
                    if model_details:
                        method += f" ({', '.join(model_details)})"
                        method += f" (agreement: {agreement_ratio:.2f})"
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
                for frame in frames:
                    try:
                        frame_array = np.expand_dims(frame, axis=0)
                        predictions = []
                        for model in self.models:
                            try:
                                pred = model.predict(frame_array, verbose=0)
                                pred_value = float(pred[0][0])
                                if 0 <= pred_value <= 1:
                                    predictions.append(pred_value)
                            except Exception:
                                pass
                        if predictions:
                            frame_predictions.append(float(np.mean(predictions)))
                    except Exception:
                        continue
                if frame_predictions:
                    confidence = float(np.mean(frame_predictions))
                    agreement_count = sum(1 for p in frame_predictions if (p > self.detection_threshold) == (confidence > self.detection_threshold))
                    agreement_ratio = agreement_count / len(frame_predictions) if frame_predictions else 0
                    if agreement_ratio >= self.min_model_agreement:
                        confidence = min(1.0, confidence * self.confidence_boost_factor)
                    is_deepfake = confidence > self.detection_threshold
                    method = f"video_{len(frame_predictions)}_frames (frame_agreement: {agreement_ratio:.2f})"
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

    def set_detection_parameters(self, threshold=0.5, min_agreement=0.6, confidence_boost=1.2):
        self.detection_threshold = max(0.0, min(1.0, threshold))
        self.min_model_agreement = max(0.0, min(1.0, min_agreement))
        self.confidence_boost_factor = max(1.0, confidence_boost)


def get_detector(force_fallback=False, model_paths=None):
    return DeepfakeDetector(force_fallback=force_fallback, model_paths=model_paths)
