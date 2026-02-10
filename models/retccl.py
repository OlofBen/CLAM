import os
import torch
import torch.nn as nn
from torchvision import models

# -------------------------------------------------------------------------
# Set the path to your downloaded RetCCL weights here
# -------------------------------------------------------------------------
RETCCL_WEIGHTS_PATH = 'workspace/models/retccl.pth'
# -------------------------------------------------------------------------

class RetCCLEncoder(nn.Module):
	"""
	Wraps a ResNet50 backbone with RetCCL pretrained weights.
	Discards the FC layer to return 2048-dim features.
	"""
	def __init__(self, weights_path=None):
		super().__init__()
		# 1. Initialize standard ResNet50
		self.model = models.resnet50(weights=None) 
		self.model.fc = nn.Identity() # Replace classification head with Identity

		# 2. Load RetCCL Weights
		if weights_path is None:
			weights_path = RETCCL_WEIGHTS_PATH

		if not os.path.exists(weights_path):
			raise FileNotFoundError(f"RetCCL weights not found at: {weights_path}")

		print(f"Loading RetCCL weights from {weights_path}...")
		
		try:
			# SECURE LOAD: explicitly disallow arbitrary code execution
			state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
		except Exception as e:
			# If the file contains malicious code or complex objects, this will likely fail safely
			raise RuntimeError(f"Failed to load weights safely. The file might be corrupted or malicious.\nError: {e}")

		# 3. Clean state_dict keys (handle DataParallel 'module.' prefix)
		new_state_dict = {}
		for k, v in state_dict.items():
			name = k.replace("module.", "") 
			new_state_dict[name] = v

		# 4. Load weights
		missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)

		real_missing = [k for k in missing if not k.startswith('fc.')]
		if real_missing:
			print(f"Warning: Missing keys during RetCCL load: {real_missing}")

	def forward(self, x):
		return self.model(x)