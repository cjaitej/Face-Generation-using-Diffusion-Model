"""Evaluation metrics for the trained model: FID, KID and Inception Score.

    python metrics.py --checkpoint models/faceforge_checkpoint.pth.tar --num-samples 2048

* FID  (lower better) -- distance between InceptionV3 feature distributions of real vs generated
  images. The standard headline metric for image generation.
* KID  (lower better) -- same idea via kernel MMD. Unlike FID it is an *unbiased* estimator, so
  it stays trustworthy at the sample counts that are actually affordable here. Prefer it when
  comparing runs below ~10k samples.
* IS   (higher better) -- per-image class confidence x class diversity. Weak for faces, since
  ImageNet classes don't describe them well, but free to compute alongside the others.

On sample count: FID is biased upward when N is small, and the bias shifts with N -- so only
compare FID values computed at the SAME --num-samples. Papers typically use 10k-50k. KID does
not have this problem, which is why it is included.

Generation is conditioned on attribute vectors drawn from the real images being compared
against, so the two sets share a conditioning distribution and the metrics reflect image
quality rather than a mismatch in which attribute combos were requested.
"""
import argparse
import json
import math
import os
from datetime import datetime

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from model import Diffusion
from utils import get_data


def _json_safe(value):
    """NaN/Inf are not valid JSON -- strict parsers reject them. A single-split Inception Score
    has no spread to report, so record that as null rather than writing an unparseable file."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def collect_real(args, num_samples, fid, kid, inception=None, as_real=True, loader=None,
                 desc='real images'):
    """Feed real images to the metrics and return the attribute vectors that came with them.

    `as_real=False` routes them to the metrics' *generated* side instead, which is what the
    real-vs-real baseline uses. Pass an existing `loader` to keep consuming where a previous
    call stopped, so the two halves of that baseline never share images.
    """
    if loader is None:
        loader = iter(get_data(args))
    conditional = bool(getattr(args, 'attr_file', None))
    attributes, seen = [], 0
    with tqdm(total=num_samples, desc=desc, dynamic_ncols=True) as pbar:
        for batch in loader:
            images, attrs = batch if conditional else (batch, None)
            images = images[:num_samples - seen]
            # get_data yields [-1, 1] floats; the metrics want uint8 [0, 255].
            images_uint8 = ((images.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
            fid.update(images_uint8.to(fid.device), real=as_real)
            kid.update(images_uint8.to(kid.device), real=as_real)
            if inception is not None:
                inception.update(images_uint8.to(inception.device))
            if conditional:
                attributes.append(attrs[:num_samples - seen])
            seen += images.shape[0]
            pbar.update(images.shape[0])
            if seen >= num_samples:
                break
    if seen < num_samples:
        raise ValueError(f'Only {seen} real images available but {num_samples} were needed. '
                         f'Lower --num-samples or point --dataset-path at more data.')
    return (torch.cat(attributes) if conditional else None), loader


def collect_generated(diffusion, model, args, attributes, fid, kid, inception):
    generated = 0
    with tqdm(total=args.num_samples, desc='generated images', dynamic_ncols=True) as pbar:
        while generated < args.num_samples:
            n = min(args.batch_size, args.num_samples - generated)
            attrs = attributes[generated:generated + n] if attributes is not None else None
            # Vary the seed per batch so the batches aren't all the same noise, but keep it
            # derived from --seed so the whole evaluation is reproducible.
            batch_seed = None if args.seed is None else args.seed + generated
            if args.ddim_steps:
                images = diffusion.sample_ddim(model, n, attributes=attrs, seed=batch_seed,
                                               guidance_scale=args.guidance_scale,
                                               ddim_steps=args.ddim_steps)
            else:
                images = diffusion.sample(model, n, attributes=attrs, seed=batch_seed,
                                          guidance_scale=args.guidance_scale)
            fid.update(images.to(fid.device), real=False)
            kid.update(images.to(kid.device), real=False)
            inception.update(images.to(inception.device))
            generated += n
            pbar.update(n)


def evaluate(args):
    device = torch.device(args.device)
    if device.type == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('WARNING: running on CPU. Inception feature extraction and sampling will both be '
              'very slow here -- pass --device cuda (or check torch.cuda.is_available()) if a '
              'GPU is present.')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = (checkpoint.get('ema_model') or checkpoint['model']).to(device).eval()
    args.image_size = checkpoint.get('image_size') or args.image_size
    schedule = checkpoint.get('schedule') or 'cosine'
    print(f'Checkpoint epoch {checkpoint["epoch"]} | {args.image_size}x{args.image_size} | '
          f'{schedule} schedule | conditional={bool(model.num_attributes)}')

    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.image_size,
                          device=device, schedule=schedule)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    # KID averages over random subsets. Each subset has to be a proper subset for that to mean
    # anything -- at subset_size == num_samples every draw is the identical full set and the
    # reported std collapses to exactly 0. Half the sample count (capped at the usual 1000)
    # keeps the subsets genuinely different so the +/- is a real spread.
    kid = KernelInceptionDistance(feature=2048,
                                  subset_size=min(1000, max(2, args.num_samples // 2))).to(device)
    inception = InceptionScore(splits=min(10, max(1, args.num_samples // 100))).to(device)

    if not model.num_attributes:
        args.attr_file = None
    attributes, loader = collect_real(args, args.num_samples, fid, kid)
    if args.real_baseline:
        # Score real images against *other* real images. Nothing generated is involved, so the
        # result is the floor imposed by the sample count alone -- the best a perfect generator
        # could possibly score here. Subtract it mentally from the model's FID before judging.
        collect_real(args, args.num_samples, fid, kid, inception=inception, as_real=False,
                     loader=loader, desc='real images (held-out half)')
    else:
        collect_generated(diffusion, model, args, attributes, fid, kid, inception)

    # KID resamples many subsets, so this step is slow enough to look like a hang without a bar.
    with tqdm(total=3, desc='computing metrics', dynamic_ncols=True) as pbar:
        fid_value = float(fid.compute())
        pbar.update(1)
        kid_mean, kid_std = kid.compute()
        pbar.update(1)
        is_mean, is_std = inception.compute()
        pbar.update(1)

    if args.real_baseline:
        sampler = 'REAL-VS-REAL BASELINE (no generation)'
    else:
        sampler = f'DDIM-{args.ddim_steps}' if args.ddim_steps else f'full-{args.noise_steps}'
    print(f'\n{"=" * 56}\n'
          f'  samples        : {args.num_samples}\n'
          f'  sampler        : {sampler}\n'
          f'  guidance scale : {args.guidance_scale}\n'
          f'  FID            : {fid_value:.3f}                (lower is better)\n'
          f'  KID            : {float(kid_mean):.5f} +/- {float(kid_std):.5f}  (lower is better)\n'
          f'  Inception Score: {float(is_mean):.3f} +/- {float(is_std):.3f}    (higher is better)\n'
          f'{"=" * 56}')
    if args.real_baseline:
        print('This is the floor at this sample count: real images scored against other real\n'
              'images. A perfect generator could not beat it. Judge the model against this\n'
              'number, not against 0.')
    else:
        print('Compare FID only against runs using the same --num-samples; KID is unbiased and\n'
              'safe to compare across sample counts. Run --real-baseline at the same\n'
              '--num-samples to see how much of this FID is small-sample bias rather than model\n'
              'quality.')

    # Everything needed to interpret the numbers later is recorded alongside them -- a bare FID
    # is meaningless without the sample count, sampler and guidance scale it was measured at.
    record = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': checkpoint['epoch'],
        'image_size': args.image_size,
        'schedule': schedule,
        'conditional': bool(model.num_attributes),
        'num_samples': args.num_samples,
        'sampler': sampler,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
        'fid': _json_safe(fid_value),
        'kid_mean': _json_safe(float(kid_mean)),
        'kid_std': _json_safe(float(kid_std)),
        'inception_score_mean': _json_safe(float(is_mean)),
        'inception_score_std': _json_safe(float(is_std)),
    }
    if args.output:
        directory = os.path.dirname(args.output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Appended as JSON Lines so repeated runs (different epochs, samplers, guidance scales)
        # accumulate into one comparable table instead of overwriting each other.
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(f'Appended results to {args.output}')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', default='models/faceforge_checkpoint.pth.tar')
    parser.add_argument('--dataset-path', default='data_set.txt')
    parser.add_argument('--attr-file', default='list_attr_celeba.csv',
                        help='set to "none" to evaluate unconditionally')
    parser.add_argument('--num-samples', type=int, default=2048,
                        help='real and generated images to compare; 10k+ for a headline FID')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--ddim-steps', type=int, default=0,
                        help='0 (default) = full reverse process, i.e. the sampler the model was '
                             'actually trained for -- this is the number to report. Set e.g. 50 '
                             'to use DDIM instead: ~20x faster, but it is an approximation and '
                             'will read as a worse FID than the model deserves.')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                        help='1.0 = none. Higher improves attribute adherence but usually '
                             'worsens FID by reducing diversity')
    parser.add_argument('--noise-steps', type=int, default=1000)
    parser.add_argument('--image-size', type=int, default=128, help='overridden by the checkpoint')
    parser.add_argument('--center-crop', type=int, default=178)
    parser.add_argument('--num-workers', type=int, default=0,
                        help='0 by default: the attribute table is a ~68MB dict of per-file '
                             'tensors, and on Windows (spawn, not fork) every worker pays ~20s '
                             'to unpickle it before the first batch. Sampling dominates runtime '
                             'here anyway, so workers buy nothing. Raise it on Linux if reading '
                             'images ever becomes the bottleneck.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', default='results/metrics.jsonl',
                        help='JSON Lines file that runs are appended to; "" to disable saving')
    parser.add_argument('--real-baseline', action='store_true',
                        help='score real images against a disjoint set of real images instead of '
                             'generating. Gives the FID floor at this --num-samples: FID is '
                             'biased upward when N is small, so this is what a *perfect* '
                             'generator would score. Needs 2x --num-samples real images.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.attr_file and args.attr_file.lower() == 'none':
        args.attr_file = None
    # get_data reads these off args too.
    args.random_flip = False       # compare against the real distribution, not an augmented one
    args.pin_memory = torch.cuda.is_available()
    args.samples_per_epoch = None
    evaluate(args)
