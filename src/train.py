import torch
from generator import Generator
from discriminator import Discriminator
from dataloader import get_dataloader
from config import Config
import os

# Initialize models
generator = Generator(Config.latent_dim, Config.img_channels).to(Config.device)
discriminator = Discriminator(Config.img_channels).to(Config.device)

# Training loop (similar to previous code, but with updated paths)
# ... (full training code as before)