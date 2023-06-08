import torch
from model import Diffusion, UNet
import torch.optim as optim
import torch.nn as nn
import argparse
from utils import *


data_folder = 'data_set.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    checkpoint = 'v7_adam_check_point_diffusion.pth.tar'
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    if checkpoint == None:
        model = UNet(input_shape=(3, 64, 64), output_shape=(3, 64, 64)).to(device)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f'\nLoaded checkpoint from epoch {start_epoch}.\n')
        model = checkpoint['model']
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    l = len(dataloader)
    sampled_images = diffusion.sample(model, n=100)
    save_images(sampled_images, os.path.join("results", "final", f"{2}.jpg"))
    print("Samples Saved.")
    for epoch in range(start_epoch, args.epochs):
        print(f"Starting epoch {epoch}:")
        for i, images in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                print("=", end="")

        if epoch%5 == 0:
            sampled_images = diffusion.sample(model, n=10)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        save_checkpoint(epoch, model, optimizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 1001
    args.batch_size = 100
    args.image_size = 64
    args.dataset_path = "data_set.txt"
    args.device = "cuda"
    args.lr = 1e-4
    train(args)