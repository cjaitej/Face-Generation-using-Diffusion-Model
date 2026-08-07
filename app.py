import os

import torch
import gradio as gr

from model import Diffusion
from dataset import SELECTED_ATTRIBUTES, build_attribute_vector

# Prefers the small weights-only export produced by export_model.py (~85MB, no pickled module),
# falling back to a full training checkpoint (~341MB) for local use. Overridable so a container
# can point at a mounted volume without a rebuild.
CHECKPOINT_PATH = os.environ.get('FACEFORGE_CHECKPOINT', 'models/faceforge_serving.pt')

# The model only knows these 10 of CelebA's 40 attributes -- it was never trained with the
# others, so there is no "requested attribute" beyond this list.
# Male and Young are each a single binary CelebA attribute (1 = the named state, 0 = its
# opposite), so the UI offers them as Gender/Age choices rather than a single ambiguous
# checkbox where the "off" state (Female / Old) would otherwise never be visible.
# The four hair attributes are mutually exclusive in reality (a face has one hair state), so
# they're a single radio choice too, with an explicit "unspecified" option since real photos
# can legitimately have none of the four set.
GENDER_CHOICES = ['Female', 'Male']
AGE_CHOICES = ['Old', 'Young']
HAIR_ATTRIBUTES = {"Black_Hair", "Blond_Hair", "Brown_Hair", "Bald"}
HAIR_CHOICES = ['Unspecified'] + [a for a in SELECTED_ATTRIBUTES if a in HAIR_ATTRIBUTES]
OTHER_ATTRIBUTES = [a for a in SELECTED_ATTRIBUTES
                    if a not in HAIR_ATTRIBUTES and a not in ('Male', 'Young')]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print('Running on CPU. Fast (DDIM) sampling is usable here -- roughly 45s for 12 images -- '
          'but full 1000-step sampling takes ~15 minutes and is effectively GPU-only.')
else:
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')

bundle = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
if 'state_dict' in bundle:  # weights-only export from export_model.py
    from export_model import load_model
    model, bundle = load_model(CHECKPOINT_PATH, device=device)
else:                        # full training checkpoint; EMA weights sample cleaner
    model = (bundle.get('ema_model') or bundle['model']).to(device).eval()
image_size = bundle.get('image_size') or 128
schedule = bundle.get('schedule') or 'cosine'
epoch = bundle.get('epoch')
diffusion = Diffusion(img_size=image_size, device=device, schedule=schedule)
print(f'Loaded {CHECKPOINT_PATH} | epoch {epoch} | {image_size}x{image_size} | {schedule} schedule '
      f'| conditional={bool(model.num_attributes)}')


@torch.no_grad()
def generate(gender, age, hair, other_attrs, num_images, guidance_scale, seed, mode,
            progress=gr.Progress(track_tqdm=True)):
    num_images = int(num_images)
    attributes = None
    if model.num_attributes:
        requested = {name: 1 for name in other_attrs}
        requested['Male'] = 1 if gender == 'Male' else 0
        requested['Young'] = 1 if age == 'Young' else 0
        if hair != 'Unspecified':
            requested[hair] = 1
        attributes = build_attribute_vector(requested, num_images).to(device)

    seed_value = int(seed) if seed not in (None, '') else None
    if mode.startswith('Fast'):
        images = diffusion.sample_ddim(model, num_images, attributes=attributes,
                                       seed=seed_value, guidance_scale=guidance_scale, ddim_steps=50)
    else:
        images = diffusion.sample(model, num_images, attributes=attributes,
                                  seed=seed_value, guidance_scale=guidance_scale)
    return [img.permute(1, 2, 0).cpu().numpy() for img in images]


with gr.Blocks(title='FaceForge') as demo:
    gr.Markdown(
        '# FaceForge — Conditional Face Generator\n'
        f'Checkpoint epoch {epoch}, {image_size}x{image_size}.\n\n'
        '**Attributes this model supports** (trained on 10 of CelebA\'s 40 attributes — '
        'requesting anything else isn\'t possible without retraining):\n'
        f'- Gender: {", ".join(GENDER_CHOICES)}\n'
        f'- Age: {", ".join(AGE_CHOICES)}\n'
        f'- Hair: {", ".join(HAIR_CHOICES)}\n'
        f'- Other: {", ".join(OTHER_ATTRIBUTES)}'
    )
    with gr.Row():
        with gr.Column(scale=1):
            gender = gr.Radio(choices=GENDER_CHOICES, value='Female', label='Gender')
            age = gr.Radio(choices=AGE_CHOICES, value='Young', label='Age')
            hair = gr.Radio(choices=HAIR_CHOICES, value='Unspecified', label='Hair')
            other_attrs = gr.CheckboxGroup(choices=OTHER_ATTRIBUTES, label='Other attributes')
            num_images = gr.Slider(10, 20, value=12, step=1, label='Number of images')
            guidance_scale = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label='Guidance scale',
                                       info='1.0 = no guidance; higher pushes harder toward the selected attributes')
            seed = gr.Number(label='Seed (optional)', precision=0)
            mode = gr.Radio(choices=['Fast (DDIM, ~50 steps)', 'High quality (Full, 1000 steps)'],
                            value='Fast (DDIM, ~50 steps)', label='Sampling mode',
                            info='Fast is ~20x quicker and good for exploring; switch to High quality for a final batch')
            generate_btn = gr.Button('Generate', variant='primary')
        with gr.Column(scale=2):
            gallery = gr.Gallery(label='Generated faces', columns=5, height='auto')

    generate_btn.click(generate, inputs=[gender, age, hair, other_attrs, num_images, guidance_scale, seed, mode],
                       outputs=gallery)

if __name__ == '__main__':
    # 0.0.0.0 so the container is reachable from outside it -- Gradio's 127.0.0.1 default is
    # unroutable behind Azure's front end. Azure App Service injects the port to listen on.
    demo.queue().launch(server_name='0.0.0.0',
                        server_port=int(os.environ.get('PORT', os.environ.get('WEBSITES_PORT', 7860))))
