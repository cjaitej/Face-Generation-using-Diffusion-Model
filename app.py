import torch
import gradio as gr

from model import Diffusion
from dataset import SELECTED_ATTRIBUTES, build_attribute_vector

CHECKPOINT_PATH = 'models/faceforge_checkpoint.pth.tar'
# These four are mutually exclusive in reality (a face has one hair state), so the UI offers
# them as a single radio choice instead of independent checkboxes.
HAIR_ATTRIBUTES = {"Black_Hair", "Blond_Hair", "Brown_Hair", "Bald"}
NON_HAIR_ATTRIBUTES = [a for a in SELECTED_ATTRIBUTES if a not in HAIR_ATTRIBUTES]
HAIR_CHOICES = ['None'] + [a for a in SELECTED_ATTRIBUTES if a in HAIR_ATTRIBUTES]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
# Prefer the EMA weights -- they generate noticeably cleaner samples than the live ones.
model = (checkpoint.get('ema_model') or checkpoint['model']).to(device).eval()
image_size = checkpoint.get('image_size') or 128
schedule = checkpoint.get('schedule') or 'cosine'
diffusion = Diffusion(img_size=image_size, device=device, schedule=schedule)
print(f'Loaded checkpoint epoch {checkpoint["epoch"]} | {image_size}x{image_size} | {schedule} schedule '
      f'| conditional={bool(model.num_attributes)}')


@torch.no_grad()
def generate(selected_attrs, hair, num_images, guidance_scale, seed, mode, progress=gr.Progress(track_tqdm=True)):
    num_images = int(num_images)
    attributes = None
    if model.num_attributes:
        requested = {name: 1 for name in selected_attrs}
        if hair != 'None':
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
        f'Checkpoint epoch {checkpoint["epoch"]}, {image_size}x{image_size}. '
        'Pick attributes and click Generate.'
    )
    with gr.Row():
        with gr.Column(scale=1):
            attr_boxes = gr.CheckboxGroup(choices=NON_HAIR_ATTRIBUTES, label='Attributes')
            hair = gr.Radio(choices=HAIR_CHOICES, value='None', label='Hair')
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

    generate_btn.click(generate, inputs=[attr_boxes, hair, num_images, guidance_scale, seed, mode], outputs=gallery)

if __name__ == '__main__':
    demo.launch()
