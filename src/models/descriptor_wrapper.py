import torch
import torch.nn as nn
from src.models.superpoint_backbone import SuperPoint

class DescriptorWrapper(nn.Module):
    def __init__(self, model_name="superpoint", pretrained_path=None, freeze_backbone=False):
        super().__init__()
        if model_name == "superpoint":
            self.model = SuperPoint()
            if pretrained_path:
                self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        else:
            raise ValueError("Unknown model name")

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model.forward_descriptors(x)  # returns normalized descriptor maps
