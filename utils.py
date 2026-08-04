import os
import copy
import itertools
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from dataset import FaceDataset


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    # CelebA aligned images are 178x218. Resizing straight to a square squashes every face
    # vertically by ~22%, so centre-crop to a square first. 178 keeps the full width with a
    # minimal crop; smaller values (e.g. 148) zoom in on the face and drop more background.
    center_crop = getattr(args, 'center_crop', 178)
    steps = []
    if center_crop:
        steps.append(torchvision.transforms.CenterCrop(center_crop))
    steps += [
        torchvision.transforms.Resize((args.image_size, args.image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    if getattr(args, 'random_flip', True):
        # Faces are roughly symmetric, so horizontal flips are a free 2x of effective data.
        steps.insert(-2, torchvision.transforms.RandomHorizontalFlip())
    transforms = torchvision.transforms.Compose(steps)
    dataset = FaceDataset(args.dataset_path, transforms, attr_file=getattr(args, 'attr_file', None))

    samples_per_epoch = getattr(args, 'samples_per_epoch', None)
    if samples_per_epoch and samples_per_epoch < len(dataset):
        # RandomSampler re-permutes on every __iter__ (i.e. every epoch, since DataLoader creates
        # a fresh iterator each time), so this draws a *different* random subset each epoch rather
        # than shuffling the same fixed slice forever. Caps per-epoch compute at samples_per_epoch
        # while still covering the full dataset given enough epochs.
        sampler = RandomSampler(dataset, replacement=False, num_samples=samples_per_epoch)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                num_workers=getattr(args, 'num_workers', 0),
                                pin_memory=getattr(args, 'pin_memory', False),
                                drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=getattr(args, 'num_workers', 0),
                                pin_memory=getattr(args, 'pin_memory', False),
                                drop_last=True)
    return dataloader


def unwrap(model):
    """Return the underlying module, stripping the DataParallel wrapper if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


class EMA:
    """Exponential moving average of the model weights.

    Sampling from the averaged weights instead of the live ones is one of the cheapest
    quality wins in diffusion training — the average is far less noisy than any single step.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(unwrap(model)).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = unwrap(model).state_dict()
        for key, value in self.ema_model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source[key].detach(), alpha=1 - self.decay)
            else:
                value.copy_(source[key])


def _iter_image_paths(folder, extensions):
    """Yield image paths lazily.

    Deliberately not os.walk: that builds the complete file list for a directory before yielding
    anything, so on CelebA it enumerates all 202k entries even when the caller only wants a few.
    On a network-backed filesystem (e.g. Kaggle's /kaggle/input) that is painfully slow. scandir
    streams entries instead, letting `limit` stop early. Order is filesystem order, not sorted.
    """
    stack = [folder]
    while stack:
        subdirs = []
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(extensions):
                    yield entry.path
                elif entry.is_dir():
                    subdirs.append(entry.path)
        stack.extend(subdirs)


def create_data_list(folder, output_file='data_set.txt', extensions=('.jpg', '.jpeg', '.png'),
                     limit=None, progress_every=25000):
    """Write one image path per line. `limit` caps the number of images, which is useful when
    compute is the binding constraint and more passes over less data beats one pass over all.

    Scanning 200k files on a network filesystem takes a while; `progress_every` prints as it goes
    so a slow scan is distinguishable from a hung one.
    """
    paths = _iter_image_paths(folder, extensions)
    if limit is not None:
        paths = itertools.islice(paths, limit)
    count = 0
    with open(output_file, 'w') as f:
        for path in paths:
            f.write(path + '\n')
            count += 1
            if progress_every and count % progress_every == 0:
                print(f'  ...{count} images found', flush=True)
    return count


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)




def save_checkpoint(epoch, model, optimizer, filename='checkpoint.pth.tar', ema=None,
                    image_size=None, schedule=None, scheduler=None):
    state = {'epoch': epoch,
             'model': unwrap(model),
             'optimizer_state_dict': optimizer.state_dict(),
             # Recorded so sampling can't silently run at the wrong resolution/schedule.
             'image_size': image_size,
             'schedule': schedule}
    if ema is not None:
        state['ema_model'] = ema.ema_model
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, filename)
