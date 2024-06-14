import torch
import torch.nn as nn
from model import Generator
import config
from PIL import Image
import os
import numpy as np
from torchvision.utils import save_image

files = "test_images/"
def load_model():
    gen = Generator().to(config.DEVICE)
    checkpoint = torch.load("app/gen.pth.tar", map_location=config.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])
    return gen

if __name__ == "__main__":
    gen = load_model()
    print(gen)