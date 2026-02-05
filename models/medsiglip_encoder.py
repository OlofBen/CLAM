import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText

class MedGemmaPatchEncoder(nn.Module):
    """
    Wraps the MedGemma vision tower and projector for patch-level encoding.
    This ensures that patch features are already in the 'LLM-ready' latent space
    before CLAM aggregation.
    """
    def __init__(self, model_id="google/medgemma-1.5-4b-it", device="cuda"):
        super().__init__()

        print(f"Loading MedGemma components from {model_id}...")
        # Load the full model to extract internal modules
        # We use bfloat16 to match MedGemma's native precision
        full_model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )

        # 1. The Vision Tower (MedSigLIP)
        # Extracts raw visual features from pixels
        self.vision_tower = full_model.model.vision_tower
        self.vision_tower.eval()

        # Freeze both to ensure no latent shift occurs
        for param in self.parameters():
            param.requires_grad = False

        self.device = device
        print("Vision tower and Projector extracted and frozen.")

    def forward(self, x):
        # x: [Batch, 3, 896, 896]
        with torch.inference_mode():
            # Get the raw tokens from the vision tower (usually 256 tokens)
            res = self.vision_tower(x).last_hidden_state
            # Return the mean of tokens to get a single 1024-dim patch vector
            return res.mean(dim=1) 
