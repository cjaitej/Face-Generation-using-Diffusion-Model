from model import Diffusion
import torch
from utils import *
import os

device = torch.device('cuda')
steps = 1000
checkpoint = 'v7_adam_check_point_diffusion.pth.tar'
unet_model = torch.load(checkpoint)['model']

diffusion_model = Diffusion(img_size=64, device='cuda')

unet_model = unet_model.to(device=device)
diffusion_model = diffusion_model

@torch.no_grad()
def display():
    x = diffusion_model.sample(unet_model, 100)
    save_images(x, os.path.join("results", "final", f"{3}.jpg"))

if __name__ == '__main__':
    display()