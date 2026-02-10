from transformers import AutoModel, AutoImageProcessor, AutoProcessor
import torch.nn as nn
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
import timm

from models.retccl import RetCCLEncoder
from models.medsiglip_encoder import MedGemmaPatchEncoder

class TimmEncoderWrapper(nn.Module):
    def __init__(self, model_name):
        super(TimmEncoderWrapper, self).__init__()
        # Use hf_hub prefix to pull directly from MahmoodLab/UNI
        self.model = timm.create_model(
            f"hf_hub:{model_name}", 
            pretrained=True, 
            init_values=1e-5, 
            num_classes=0,       # Removes the head so we get features
            dynamic_img_size=True # Allows for different patch sizes/resolutions
        )
        self.model.eval()

    def forward(self, x):
        # forward_features returns [B, N, D] where N includes [CLS] + patches
        # For UNI (ViT), the [CLS] token is at index 0
        features = self.model.forward_features(x)
        return features[:, 0, :]

class HFEncoderWrapper(nn.Module):
    """Wraps any HF Vision model to return only the pooled visual features."""
    def __init__(self, model_id, trust_remote_code = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    def forward(self, x):
        outputs = self.model(x)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            lhs = outputs.last_hidden_state
            # If 4D: [Batch, Channels, Height, Width] -> ResNet/CNN
            if lhs.dim() == 4:
                # Global Average Pool: reduces H and W to 1, leaving [Batch, Channels]
                return lhs.mean(dim=[2, 3])

            # If 3D: [Batch, Sequence, Dim] -> ViT/Hibou
            elif lhs.dim() == 3:
                # Return the [CLS] token at index 0
                return lhs[:, 0, :]
        else:
            return outputs[0]

def get_encoder(model_name, target_img_size=224, trust_remote_code = False):
    """
    Revised get_encoder to support HF models and RetCCL.
    Ex: 'facebook/vit-base-patch16-224' or 'microsoft/resnet-50'
    """
    # -----------------------------------------------------------
    # 1. RetCCL Specific Handler
    # -----------------------------------------------------------
    if model_name == 'retccl':
        model = RetCCLEncoder()
        
        # RetCCL uses standard ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        
        img_transforms = get_eval_transforms(
            mean=mean, 
            std=std, 
            target_img_size=target_img_size
        )
        return model, img_transforms

    # -----------------------------------------------------------
    # 2. Legacy CLAM Models (ResNet50_trunc)
    # -----------------------------------------------------------
    if model_name == 'resnet50_trunc':
        from .timm_wrapper import TimmCNNEncoder
        model = TimmCNNEncoder()
        constants = MODEL2CONSTANTS[model_name]
        return model, get_eval_transforms(
            mean=constants['mean'],
            std= constants['std'],
            target_img_size=target_img_size)

    #-----------------------------------------------------------
    # 3. MedGemma-Native MedSigLIP
    # -----------------------------------------------------------
    if model_name == 'medsiglip':
        model_id = "google/medgemma-1.5-4b-it"

        model = MedGemmaPatchEncoder(model_id=model_id)

        # We extract mean and std from the processor
        processor = AutoProcessor.from_pretrained(model_id)
        mean = processor.image_processor.image_mean
        std = processor.image_processor.image_std

        img_transforms = get_eval_transforms(
            mean=mean, 
            std=std, 
            target_img_size=target_img_size
        )
        return model, img_transforms

    # -----------------------------------------------------------
    # 4. UNI models
    # -----------------------------------------------------------
    if "UNI" in model_name:
        # --- UNI SPECIFIC LOADING ---
        model = TimmEncoderWrapper(model_name)

        # UNI uses standard ImageNet normalization
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        img_transforms = get_eval_transforms(
            mean=mean, 
            std=std, 
            target_img_size=target_img_size
        )

        print(f"Successfully loaded {model_name} via timm.")
        return model, img_transforms

    # -----------------------------------------------------------
    # 5. Hugging Face AutoModel Fallback
    # -----------------------------------------------------------
    print(f"Attempting to load {model_name} from Hugging Face...")
    try:
        model = HFEncoderWrapper(model_name, trust_remote_code)

        # Extract mean and std with safe defaults (ImageNet values)
        try:
            processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            img_proc = getattr(processor, "image_processor", processor)
            mean = getattr(img_proc, "image_mean", [0.485, 0.456, 0.406])
            std = getattr(img_proc, "image_std", [0.229, 0.224, 0.225])
            print(f"Successfully loaded processor for {model_name}")
        except Exception:
            # Fallback for models missing the config file
            print(f"No config found for {model_name}. Using default DINOv2/ImageNet transforms.")
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img_transforms = get_eval_transforms(
                mean=mean,
                std= std,
                target_img_size=target_img_size
            )

        print(f"Successfully loaded HF model: {model_name}")
        return model, img_transforms

    except Exception as e:
        print(f"HF Load failed: {e}")
        raise NotImplementedError(f'Model {model_name} not found in legacy, RetCCL, or HF Hub.')

# Legacy compatibility
def has_CONCH(): return False, ""
def has_UNI(): return False, ""
