import os
import copy
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
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


def create_data_list(folder, output_file='data_set.txt', extensions=('.jpg', '.jpeg', '.png'),
                     limit=None):
    """Write one image path per line. `limit` caps the number of images, which is useful when
    compute is the binding constraint and more passes over less data beats one pass over all."""
    count = 0
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                if not name.lower().endswith(extensions):
                    continue
                f.write(os.path.join(root, name) + '\n')
                count += 1
                if limit is not None and count >= limit:
                    return count
    return count


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)




def save_checkpoint(epoch, model, optimizer, filename='checkpoint.pth.tar', ema=None,
                    image_size=None, schedule=None):
    state = {'epoch': epoch,
             'model': unwrap(model),
             'optimizer_state_dict': optimizer.state_dict(),
             # Recorded so sampling can't silently run at the wrong resolution/schedule.
             'image_size': image_size,
             'schedule': schedule}
    if ema is not None:
        state['ema_model'] = ema.ema_model
    torch.save(state, filename)
