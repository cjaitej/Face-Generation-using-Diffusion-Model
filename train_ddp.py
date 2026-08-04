"""Multi-GPU launcher: the same config you'd set in the notebook, as a plain script.

Run it with torchrun, one process per GPU:

    torchrun --nproc_per_node=2 train_ddp.py

Defining the config here rather than in a notebook cell is what makes multi-GPU work --
torchrun launches independent processes that each execute this file, so nothing has to be
pickled out of a Jupyter kernel (which is what breaks `mp.spawn` from a notebook).
"""
import os
import torch
from train import train


class Args:
    pass


args = Args()
args.run_name = 'FaceForge_Conditional'
args.epochs = 500

# NOTE: under DDP this is the PER-GPU batch size, not the total. 256 x 2 GPUs = the same
# global batch of 512 the single-GPU run was using.
args.batch_size = 256

args.image_size = 128
args.center_crop = 178              # CelebA is 178x218 - crop square before resizing
args.random_flip = True
args.num_workers = min(4, os.cpu_count() or 2)
args.pin_memory = True

# CHECK THESE TWO PATHS -- they must match wherever the data actually lives on this machine.
args.dataset_path = 'data_set.txt'
args.attr_file = 'list_attr_celeba.csv'   # set to None to train the unconditional baseline

args.samples_per_epoch = 60000      # fresh random subset each epoch; None = full list every epoch
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.lr = 1e-4
args.use_amp = torch.cuda.is_available()
args.noise_steps = 1000
args.time_emb_dim = 256
args.dropout = 0.0
args.schedule = 'cosine'            # diffusion noise schedule (not the LR schedule)
args.min_snr_gamma = 5.0            # Min-SNR-gamma loss weighting clip (Hang et al., 2023)
args.use_ema = True
args.ema_decay = 0.999
args.cond_drop_prob = 0.1           # classifier-free guidance conditioning dropout
args.guidance_scale = 3.0           # guidance strength for the preview grids
args.sample_every = 5               # save a preview grid every N epochs
args.sample_seed = 1234             # base seed for preview randomization (offset by epoch)
args.n_eval_samples = 8             # number of randomized faces per preview grid
args.distributed = True

args.checkpoint_path = './models/faceforge_checkpoint.pth.tar'
args.resume_checkpoint = args.checkpoint_path  # set to None to start over from scratch

os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

if __name__ == '__main__':
    train(args)
