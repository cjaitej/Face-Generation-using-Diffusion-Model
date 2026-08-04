import os
import torch
from model import Diffusion, UNet
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
from utils import *
from dataset import SELECTED_ATTRIBUTES, random_attribute_batch


def train(args):
    setup_logging(args.run_name)
    device = torch.device(args.device)
    if device.type == 'cuda':
        # Every step uses the same fixed input shape (image size, batch size), so cuDNN can
        # safely benchmark and cache the fastest conv algorithm for it instead of picking a
        # generic one each time -- essentially free speedup for this training loop's static shapes.
        torch.backends.cudnn.benchmark = True
    dataloader = get_data(args)
    num_attributes = len(SELECTED_ATTRIBUTES) if getattr(args, 'attr_file', None) else 0

    resume_checkpoint = getattr(args, 'resume_checkpoint', None)
    optimizer_state = None
    scheduler_state = None
    ema_model = None
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model = checkpoint['model']
        ema_model = checkpoint.get('ema_model')
        optimizer_state = checkpoint.get('optimizer_state_dict')
        scheduler_state = checkpoint.get('scheduler_state_dict')
        start_epoch = checkpoint['epoch'] + 1
        print(f'\nLoaded checkpoint from epoch {start_epoch}.\n')
    else:
        model = UNet(input_shape=(3, args.image_size, args.image_size),
                     output_shape=(3, args.image_size, args.image_size),
                     num_attributes=num_attributes,
                     time_emb_dim=getattr(args, 'time_emb_dim', 256),
                     dropout=getattr(args, 'dropout', 0.0))
        start_epoch = 0

    model = model.to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    ema = None
    if getattr(args, 'use_ema', True):
        ema = EMA(model, decay=getattr(args, 'ema_decay', 0.999))
        if ema_model is not None:  # restore the averaged weights when resuming
            ema.ema_model = ema_model
        ema.ema_model = ema.ema_model.to(device)

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'Using DataParallel across {torch.cuda.device_count()} GPUs')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Cosine decay over the full run so resuming with a raised `epochs` target would shift the
    # decay horizon -- set epochs to the value you actually intend to train to before resuming.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        # load_state_dict restores the scheduler's own counters but doesn't push the resumed LR
        # back into the optimizer -- without this it would silently train the next epoch at the
        # original (pre-decay) lr until the following scheduler.step() corrected it.
        for param_group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            param_group['lr'] = lr

    mse = nn.MSELoss(reduction='none')
    diffusion = Diffusion(noise_steps=getattr(args, 'noise_steps', 1000),
                          img_size=args.image_size, device=device,
                          schedule=getattr(args, 'schedule', 'cosine'))
    min_snr_gamma = getattr(args, 'min_snr_gamma', 5.0)

    use_amp = getattr(args, 'use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    checkpoint_path = getattr(args, 'checkpoint_path', 'checkpoint.pth.tar')
    # Fraction of samples trained with the conditioning replaced by the learned null token.
    # This is what makes classifier-free guidance possible at sampling time.
    cond_drop_prob = getattr(args, 'cond_drop_prob', 0.1) if num_attributes else 0.0

    # Attribute combos + noise are freshly randomized each preview (seeded off the epoch number,
    # so a given epoch is reproducible across runs) -- shows the model's general range instead of
    # tracking the same fixed set of faces every time.
    sample_every = getattr(args, 'sample_every', 5)
    sample_seed = getattr(args, 'sample_seed', 1234)
    guidance_scale = getattr(args, 'guidance_scale', 3.0)
    n_eval = getattr(args, 'n_eval_samples', 8)

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"epoch {epoch}/{args.epochs - 1}", dynamic_ncols=True)
        for i, batch in enumerate(pbar):
            if num_attributes:
                images, attributes = batch
                attributes = attributes.to(device)
            else:
                images, attributes = batch, None
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                predicted_noise = model(x_t, t, attributes, cond_drop_prob)
                per_sample_loss = mse(noise, predicted_noise).mean(dim=[1, 2, 3])
                loss = (diffusion.min_snr_weights(t, min_snr_gamma) * per_sample_loss).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            loss_value = loss.item()  # one CUDA sync per step instead of two
            running_loss += loss_value
            pbar.set_postfix(loss=f"{loss_value:.6f}", avg=f"{running_loss / (i + 1):.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        pbar.close()
        scheduler.step()
        print(f"epoch {epoch} avg loss: {running_loss / max(len(dataloader), 1):.4f}")

        if epoch % sample_every == 0:
            sampling_model = ema.ema_model if ema is not None else unwrap(model)
            epoch_seed = sample_seed + epoch
            eval_attributes = random_attribute_batch(n_eval, seed=epoch_seed).to(device) if num_attributes else None
            sampled_images = diffusion.sample(sampling_model, n=n_eval, attributes=eval_attributes,
                                              seed=epoch_seed, guidance_scale=guidance_scale)
            sample_path = os.path.join("results", args.run_name, f"epoch_{epoch:04d}.jpg")
            save_images(sampled_images, sample_path)
            print(f"Saved samples to {sample_path}")
        save_checkpoint(epoch, model, optimizer, filename=checkpoint_path, ema=ema,
                        image_size=args.image_size, schedule=getattr(args, 'schedule', 'cosine'),
                        scheduler=scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "FaceForge_Conditional"
    args.epochs = 1001
    args.batch_size = 32
    args.image_size = 128
    args.center_crop = 178         # CelebA is 178x218; crop square before resizing
    args.random_flip = True
    args.num_workers = 4
    args.pin_memory = torch.cuda.is_available()
    args.dataset_path = "data_set.txt"
    args.samples_per_epoch = None  # e.g. 60000: cap per-epoch batches, redrawn randomly each epoch
    args.attr_file = "list_attr_celeba.csv"  # set to None to train the original unconditional model
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = 1e-4
    args.resume_checkpoint = None  # set to a checkpoint path to resume training
    args.checkpoint_path = "checkpoint.pth.tar"
    args.use_amp = torch.cuda.is_available()
    args.noise_steps = 1000
    args.time_emb_dim = 256
    args.dropout = 0.0
    args.schedule = "cosine"       # or "linear" for the original schedule
    args.min_snr_gamma = 5.0       # Min-SNR-gamma loss weighting clip (Hang et al., 2023)
    args.use_ema = True
    args.ema_decay = 0.999
    args.cond_drop_prob = 0.1      # classifier-free guidance conditioning dropout
    args.guidance_scale = 3.0      # guidance strength used for the preview grids
    args.sample_every = 5          # save a preview grid every N epochs
    args.sample_seed = 1234        # base seed for preview randomization (offset by epoch)
    args.n_eval_samples = 8        # number of randomized faces per preview grid
    train(args)
