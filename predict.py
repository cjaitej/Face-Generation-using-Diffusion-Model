from model import Diffusion
import torch
from utils import *
from dataset import SELECTED_ATTRIBUTES, build_attribute_vector
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = 'checkpoint.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Prefer the EMA weights — they generate noticeably cleaner samples than the live ones.
unet_model = checkpoint.get('ema_model') or checkpoint['model']
unet_model = unet_model.to(device)

# Resolution and schedule come from the checkpoint so sampling always matches training.
IMAGE_SIZE = checkpoint.get('image_size') or 128
SCHEDULE = checkpoint.get('schedule') or 'cosine'
print(f'checkpoint epoch {checkpoint["epoch"]} | {IMAGE_SIZE}x{IMAGE_SIZE} | {SCHEDULE} schedule')

diffusion_model = Diffusion(img_size=IMAGE_SIZE, device=device, schedule=SCHEDULE)

# Attribute values requested for generation, e.g. {"Male": 1, "Smiling": 1, "Black_Hair": 1}.
REQUESTED_ATTRIBUTES = {"Male": 0, "Young": 1, "Smiling": 1, "Black_Hair": 1}

SEED = None            # set to an int to reproduce an identical batch of faces
GUIDANCE_SCALE = 3.0   # >1 pushes samples harder towards the requested attributes


@torch.no_grad()
def display():
    n = 100
    attributes = build_attribute_vector(REQUESTED_ATTRIBUTES, n) if unet_model.num_attributes else None
    x = diffusion_model.sample(unet_model, n, attributes=attributes,
                               seed=SEED, guidance_scale=GUIDANCE_SCALE)
    os.makedirs("results", exist_ok=True)
    save_images(x, os.path.join("results", "generated.jpg"))


if __name__ == '__main__':
    display()
