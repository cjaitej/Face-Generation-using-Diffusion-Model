import os
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model import Diffusion, UNet
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
from utils import *
from dataset import SELECTED_ATTRIBUTES, random_attribute_batch


def train_worker(rank, world_size, args, local_rank=None):
    """Runs the full training loop in one process. With world_size > 1 this is one of several
    processes (spawned by `train()` or launched by torchrun), each driving its own GPU via
    DistributedDataParallel -- gradients are synchronized (all-reduced) automatically on every
    backward() call, so this function doesn't need to do anything special for that. Logging,
    sampling and checkpointing only happen on rank 0 to avoid duplicate work and file-write
    races between processes.

    `local_rank` is the GPU index on this machine; it differs from `rank` only in multi-node
    runs, so it defaults to `rank`.
    """
    is_main = rank == 0
    distributed = world_size > 1
    if local_rank is None:
        local_rank = rank
    if distributed:
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        os.environ.setdefault('USE_LIBUV', '0')  # some torch builds (notably Windows) lack libuv support
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        if not dist.is_initialized():
            dist.init_process_group(backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    if is_main:
        setup_logging(args.run_name)
    if device.type == 'cuda':
        # Every step uses the same fixed input shape (image size, batch size), so cuDNN can
        # safely benchmark and cache the fastest conv algorithm for it instead of picking a
        # generic one each time -- essentially free speedup for this training loop's static shapes.
        torch.backends.cudnn.benchmark = True
    dataloader = get_data(args, rank=rank, world_size=world_size)
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
        if is_main:
            print(f'\nLoaded checkpoint from epoch {start_epoch}.\n')
    else:
        model = UNet(input_shape=(3, args.image_size, args.image_size),
                     output_shape=(3, args.image_size, args.image_size),
                     num_attributes=num_attributes,
                     time_emb_dim=getattr(args, 'time_emb_dim', 256),
                     dropout=getattr(args, 'dropout', 0.0))
        start_epoch = 0

    model = model.to(device)
    if is_main:
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    ema = None
    if getattr(args, 'use_ema', True):
        ema = EMA(model, decay=getattr(args, 'ema_decay', 0.999))
        if ema_model is not None:  # restore the averaged weights when resuming
            ema.ema_model = ema_model
        ema.ema_model = ema.ema_model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)
        if is_main:
            print(f'Using DistributedDataParallel across {world_size} processes')

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
    elif start_epoch > 0:
        # Resuming from a checkpoint saved before the scheduler existed (no scheduler_state_dict).
        # Without this, the freshly-constructed scheduler would restart its cosine decay from 0
        # instead of picking up at the real epoch -- fast-forward it to the correct point instead.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # expected: step() called without an interleaved optimizer.step() here
            for _ in range(start_epoch):
                scheduler.step()

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
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        iterator = tqdm(dataloader, desc=f"epoch {epoch}/{args.epochs - 1}", dynamic_ncols=True) if is_main else dataloader
        for i, batch in enumerate(iterator):
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

            if is_main:
                loss_value = loss.item()  # one CUDA sync per step instead of two
                running_loss += loss_value
                iterator.set_postfix(loss=f"{loss_value:.6f}", avg=f"{running_loss / (i + 1):.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        if is_main:
            iterator.close()
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

        if distributed:
            dist.barrier()  # keep ranks aligned at the epoch boundary while rank 0 samples/saves

    if distributed:
        dist.destroy_process_group()


def _running_in_notebook():
    """True inside a Jupyter/IPython kernel. mp.spawn cannot be used there: the spawn start
    method re-imports __main__ (which is the kernel, not an importable module) and pickles the
    args object by reference to __main__, so any class defined in a notebook cell fails to
    resolve in the child. Children die during startup with a bare non-zero exit code."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None and 'IPKernelApp' in get_ipython().config
    except Exception:
        return False


def train(args):
    """Entry point, supporting three launch modes:

    1. `torchrun --nproc_per_node=N train.py` -- recommended for multi-GPU. torchrun starts the
       processes itself and sets RANK/WORLD_SIZE/LOCAL_RANK, so this process is already a worker.
    2. `python train.py` on a multi-GPU box -- spawns one process per GPU via mp.spawn.
    3. Single GPU, CPU, or inside a notebook -- runs in-process.

    Set args.distributed = False to force single-process even with multiple GPUs visible.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        train_worker(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), args,
                     local_rank=int(os.environ.get('LOCAL_RANK', os.environ['RANK'])))
        return

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1 and getattr(args, 'distributed', True):
        if _running_in_notebook():
            print(f'WARNING: {world_size} GPUs are visible, but multi-GPU training cannot be '
                  'launched from a notebook (mp.spawn cannot pickle notebook-defined objects '
                  'or re-import the kernel as __main__). Falling back to a single GPU.\n'
                  '         To use all GPUs, run training as a script instead:\n'
                  f'             torchrun --nproc_per_node={world_size} train.py\n'
                  '         Set args.distributed = False to silence this warning.')
            train_worker(0, 1, args)
            return
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "FaceForge_Conditional"
    args.epochs = 1001
    # Per-process batch size under DistributedDataParallel (each GPU gets this many images per
    # step, not the total across GPUs) -- unlike the old DataParallel setup, you don't need to
    # multiply this by the GPU count yourself.
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
    args.distributed = True        # set False to force single-process even with multiple GPUs visible
    train(args)
