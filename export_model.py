"""Strip a training checkpoint down to what inference actually needs.

    python export_model.py --checkpoint models/faceforge_checkpoint_fast.pth.tar

A training checkpoint carries the live model, the EMA copy, optimizer state and scheduler state
(~325MB). Serving needs only the EMA weights, so this writes a ~85MB file instead -- worth doing
before building a container image around it.

It also switches from a pickled `nn.Module` to a plain `state_dict`. Pickled modules have to be
loaded with `weights_only=False`, which executes arbitrary code on load and silently breaks if
the class definition moves; a state_dict has neither problem.
"""
import argparse

import torch

from model import UNet


def export(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # EMA weights generate noticeably cleaner samples than the live ones.
    model = checkpoint.get('ema_model') or checkpoint['model']

    image_size = checkpoint.get('image_size') or 128
    bundle = {
        'state_dict': model.state_dict(),
        'num_attributes': model.num_attributes,
        'image_size': image_size,
        'schedule': checkpoint.get('schedule') or 'cosine',
        # Read off the built module rather than assumed, so an export stays correct if the
        # architecture defaults ever change.
        'time_emb_dim': model.time_mlp[1].in_features,
        'epoch': checkpoint.get('epoch'),
    }
    torch.save(bundle, output_path)
    return bundle


def load_model(path, device='cpu'):
    """Rebuild a UNet from an exported bundle. Safe to load: no pickled module, no code execution."""
    bundle = torch.load(path, map_location=device, weights_only=False)
    shape = (3, bundle['image_size'], bundle['image_size'])
    model = UNet(input_shape=shape, output_shape=shape,
                 num_attributes=bundle['num_attributes'],
                 time_emb_dim=bundle['time_emb_dim'])
    model.load_state_dict(bundle['state_dict'])
    return model.to(device).eval(), bundle


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', default='models/faceforge_checkpoint_fast.pth.tar')
    parser.add_argument('--output', default='models/faceforge_serving.pt')
    args = parser.parse_args()

    bundle = export(args.checkpoint, args.output)
    before = os.path.getsize(args.checkpoint) / 1e6
    after = os.path.getsize(args.output) / 1e6
    print(f'epoch {bundle["epoch"]} | {bundle["image_size"]}x{bundle["image_size"]} | '
          f'{bundle["schedule"]} | {bundle["num_attributes"]} attributes')
    print(f'{args.checkpoint}  {before:.0f} MB')
    print(f'{args.output}  {after:.0f} MB  ({before / after:.1f}x smaller)')

    # Verify the exported file actually rebuilds before it gets baked into an image.
    model, _ = load_model(args.output)
    print(f'reload OK: {sum(p.numel() for p in model.parameters()):,} params')
