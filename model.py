import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def VGG(num_classes):
    model = torchvision.models.vgg11(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[-1] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes))
    return model
