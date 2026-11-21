# -*- coding: utf-8 -*-
"""ml/model.py — 1채널 입력 ResNet18 (baseline과 동일 구조)."""
import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    old = model.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(old.weight.sum(dim=1, keepdim=True))  # RGB 가중치 합 -> 1채널
    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
