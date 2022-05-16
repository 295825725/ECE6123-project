# import torch
import torch.nn as nn
from torchvision import models

class BaseModel(nn.Module):
    """Get the base network, which is modified from ResNet50"""
    def __init__(self, IF_PRETRAINED=False):
        super(BaseModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=IF_PRETRAINED)
        self.resnet50.fc = nn.Linear(2048, 258)

    def forward(self, images):
        return self.resnet50(images)