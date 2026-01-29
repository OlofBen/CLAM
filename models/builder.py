from transformers import AutoModel, AutoImageProcessor
import torch.nn as nn
from utils.constants import MODEL2CONSTANTS

from utils.transform_utils import get_eval_transforms

class HFEncoderWrapper(nn.Module):
    """Wraps any HF Vision model to return only the pooled visual features."""
    def __init__(self, model_id):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    def forward(self, x):
        outputs = self.model(x)
        # Handle ViT/Swin (last_hidden_state) vs CLIP (image_embeds)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Return mean spatial pooling if no pooler_output exists
            return outputs.last_hidden_state.mean(dim=1)
        else:
            # Fallback for older models or different return types
            return outputs[0]

def get_encoder(model_name, target_img_size=224):
    """
    Revised get_encoder to support HF models by name.
    Ex: 'facebook/vit-base-patch16-224' or 'microsoft/resnet-50'
    """
    # 1. Check for hardcoded legacy models first
    if model_name == 'resnet50_trunc':
        from .timm_wrapper import TimmCNNEncoder
        model = TimmCNNEncoder()
        constants = MODEL2CONSTANTS[model_name]

        return model, get_eval_transforms(
            mean=constants['mean'],
            std= constants['std'],
            target_img_size=target_img_size)

    # 2. Try loading from Hugging Face
    print(f"Attempting to load {model_name} from Hugging Face...")
    try:
        # Load the processor to get normalization constants (mean/std)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = HFEncoderWrapper(model_name)

        # Get mean/std from processor if available, else use ImageNet defaults
        mean = getattr(processor, 'image_mean', [0.485, 0.456, 0.406])
        std = getattr(processor, 'image_std', [0.229, 0.224, 0.225])

        img_transforms = get_eval_transforms(
            mean=mean,
            std= std,
            target_img_size=target_img_size
        )

        print(f"Successfully loaded HF model: {model_name}")
        return model, img_transforms

    except Exception as e:
        print(f"HF Load failed: {e}")
        raise NotImplementedError(f'Model {model_name} not found in legacy or HF Hub.')

# for legacy reasons
def has_CONCH():
    # Return a dummy value to satisfy the import
    return False, ""

def has_UNI():
    # Return a dummy value to satisfy the import
    return False, ""