import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


def norm_layer(channels):
    """GroupNorm with a sensible group count (~4 channels per group, capped at 32 groups).
    Falls back to a divisor of `channels` so the layer is always constructible."""
    groups = min(32, max(1, channels // 4))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Pre-activation residual block with the timestep/attribute embedding injected between
    the two convolutions (standard DDPM placement)."""

    def __init__(self, in_c, out_c, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = norm_layer(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.norm2 = norm_layer(out_c)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x, t):
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.time_mlp(self.activation(t))[(..., ) + (None, ) * 2]
        h = self.conv2(self.dropout(self.activation(self.norm2(h))))
        return h + self.skip(x)


class BottleneckResBlock(nn.Module):
    """Pre-activation residual block, ResNet-v2 bottleneck style: squeeze to a narrow channel
    count, do the 3x3 spatial mixing there, then expand back out.

    A plain 3x3-then-3x3 ResBlock's cost scales with out_c^2, which is fine at 32-64 channels
    but dominates the parameter budget once channels reach the hundreds (e.g. a 1024-channel
    block is ~85% of the whole UNet). Squeezing to out_c/reduction before the 3x3 keeps the
    1x1 projections at full width (so skip connections and channel-mixing capacity are
    unchanged) while cutting the expensive conv's cost by roughly reduction^2.
    """

    def __init__(self, in_c, out_c, time_emb_dim, dropout=0.0, reduction=4):
        super().__init__()
        mid_c = max(out_c // reduction, 32)
        self.norm1 = norm_layer(in_c)
        self.reduce = nn.Conv2d(in_c, mid_c, kernel_size=1)
        self.norm_mid1 = norm_layer(mid_c)
        self.conv3x3 = nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, mid_c)
        self.norm_mid2 = norm_layer(mid_c)
        self.dropout = nn.Dropout(dropout)
        self.expand = nn.Conv2d(mid_c, out_c, kernel_size=1)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x, t):
        h = self.reduce(self.activation(self.norm1(x)))
        h = self.conv3x3(self.activation(self.norm_mid1(h)))
        h = h + self.time_mlp(self.activation(t))[(..., ) + (None, ) * 2]
        h = self.expand(self.dropout(self.activation(self.norm_mid2(h))))
        return h + self.skip(x)


# Below this channel count a plain ResBlock is already cheap; at and above it the wide 3x3s
# dominate the parameter budget, so the bottleneck design is worth the extra 1x1s.
BOTTLENECK_CHANNEL_THRESHOLD = 128


def make_res_block(in_c, out_c, time_emb_dim, dropout=0.0, reduction=4):
    if out_c >= BOTTLENECK_CHANNEL_THRESHOLD:
        return BottleneckResBlock(in_c, out_c, time_emb_dim, dropout, reduction=reduction)
    return ResBlock(in_c, out_c, time_emb_dim, dropout)


class Attention(nn.Module):
    """Vectorized spatial self-attention over the H*W positions."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = norm_layer(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = self.norm(x).reshape(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.attention(seq, seq, seq, need_weights=False)
        return x + attn_out.transpose(1, 2).reshape(b, c, h, w)


class Downsample(nn.Module):
    """Strided convolution; preserves more detail than max pooling."""

    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsample + conv, which avoids the checkerboard artifacts
    that transposed convolutions produce."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class EncoderLevel(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, use_attention, dropout=0.0):
        super().__init__()
        self.res_block = make_res_block(in_c, out_c, time_emb_dim, dropout)
        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(out_c)
        self.down = Downsample(out_c)

    def forward(self, x, t):
        x = self.res_block(x, t)
        if self.use_attention:
            x = self.attention(x)
        return self.down(x), x  # (downsampled, skip)


class DecoderLevel(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, use_attention, dropout=0.0):
        super().__init__()
        self.up = Upsample(in_c, out_c)
        self.res_block = make_res_block(out_c * 2, out_c, time_emb_dim, dropout)
        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(out_c)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x, t)
        if self.use_attention:
            x = self.attention(x)
        return x


class UNet(nn.Module):
    """Conditional U-Net predicting the noise added at a given timestep.

    Attributes are embedded and summed into the timestep embedding. A learned `null_cond`
    token represents "no conditioning", which is what makes classifier-free guidance work:
    during training some samples are randomly switched to it, and at sampling time the
    conditional and unconditional predictions are extrapolated apart.
    """

    def __init__(self, input_shape, output_shape, num_attributes=0, time_emb_dim=256,
                 attention_resolutions=(16, 8), dropout=0.0):
        super(UNet, self).__init__()
        self.channels = [32, 64, 128, 256, 512]
        self.num_attributes = num_attributes

        img_size = input_shape[-1]
        # Resolution each encoder level operates at, before its downsample.
        enc_resolutions = [img_size // (2 ** i) for i in range(len(self.channels))]

        self.stem = nn.Conv2d(input_shape[0], self.channels[0], kernel_size=3, padding=1)

        enc_in = [self.channels[0]] + self.channels[:-1]
        self.encoder_list = nn.ModuleList([
            EncoderLevel(i_c, o_c, time_emb_dim, res in attention_resolutions, dropout)
            for i_c, o_c, res in zip(enc_in, self.channels, enc_resolutions)
        ])

        bottleneck_c = self.channels[-1] * 2
        self.bottleneck_in = make_res_block(self.channels[-1], bottleneck_c, time_emb_dim, dropout)
        self.bottleneck_attention = Attention(bottleneck_c)
        self.bottleneck_out = make_res_block(bottleneck_c, bottleneck_c, time_emb_dim, dropout)

        dec_out = self.channels[::-1]
        dec_in = [bottleneck_c] + dec_out[:-1]
        dec_resolutions = enc_resolutions[::-1]
        self.decoder_list = nn.ModuleList([
            DecoderLevel(i_c, o_c, time_emb_dim, res in attention_resolutions, dropout)
            for i_c, o_c, res in zip(dec_in, dec_out, dec_resolutions)
        ])

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        if num_attributes > 0:
            self.label_mlp = nn.Sequential(
                nn.Linear(num_attributes, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            self.null_cond = nn.Parameter(torch.zeros(time_emb_dim))

        self.out_norm = norm_layer(self.channels[0])
        self.out_activation = nn.SiLU()
        # No output activation: the target is noise ~ N(0, 1), which is signed and unbounded.
        self.output = nn.Conv2d(self.channels[0], output_shape[0], kernel_size=1)
        # Zero-init so the model starts by predicting zero noise, which stabilises early training.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _conditioning(self, attributes, batch, cond_drop_prob, device):
        null_cond = self.null_cond[None, :].expand(batch, -1)
        if attributes is None:
            return null_cond
        cond = self.label_mlp(attributes)
        if cond_drop_prob > 0:
            drop = torch.rand(batch, device=device) < cond_drop_prob
            cond = torch.where(drop[:, None], null_cond.to(cond.dtype), cond)
        return cond

    def forward(self, x, timestep, attributes=None, cond_drop_prob=0.0):
        t = self.time_mlp(timestep)
        if self.num_attributes > 0:
            t = t + self._conditioning(attributes, x.shape[0], cond_drop_prob, x.device)

        x = self.stem(x)

        skips = []
        for encoder in self.encoder_list:
            x, skip = encoder(x, t)
            skips.append(skip)

        x = self.bottleneck_in(x, t)
        x = self.bottleneck_attention(x)
        x = self.bottleneck_out(x, t)

        for decoder, skip in zip(self.decoder_list, skips[::-1]):
            x = decoder(x, skip, t)

        return self.output(self.out_activation(self.out_norm(x)))


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64,
                 device="cuda", schedule='cosine'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule = schedule

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        if self.schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        # Cosine schedule (Nichol & Dhariwal); destroys information more gradually than
        # linear, which noticeably helps at low resolutions like 64x64.
        s = 0.008
        steps = torch.linspace(0, self.noise_steps, self.noise_steps + 1) / self.noise_steps
        alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=0.999)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def min_snr_weights(self, t, gamma=5.0):
        """Min-SNR-gamma loss weighting (Hang et al., ICCV 2023). Downweights low-noise/
        high-SNR timesteps, which have a hard floor on achievable loss (recovering the exact
        noise draw from an almost-clean image is bounded by the image's own pixel-level
        randomness) and so would otherwise dominate the gradient without much to show for it."""
        snr = self.alpha_hat[t] / (1 - self.alpha_hat[t])
        return torch.clamp(snr, max=gamma) / snr

    def sample(self, model, n, attributes=None, seed=None, guidance_scale=1.0):
        """Generate n images.

        `seed` fixes the noise so runs are reproducible and samples saved at different epochs
        stay directly comparable. `guidance_scale` > 1 applies classifier-free guidance, pushing
        samples further towards the requested attributes (typical range 1.5-5).
        """
        model.eval()
        if attributes is not None:
            attributes = attributes.to(self.device)
        use_guidance = attributes is not None and guidance_scale != 1.0

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device, generator=generator)
            for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1,
                          desc=f"sampling {n} images", dynamic_ncols=True, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                if use_guidance:
                    predicted_noise = model(x, t, attributes)
                    uncond_noise = model(x, t, None)
                    predicted_noise = uncond_noise + guidance_scale * (predicted_noise - uncond_noise)
                else:
                    predicted_noise = model(x, t, attributes)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Clip the implied x0 estimate to the valid data range at every step. The
                # reconstruction below divides by sqrt(alpha_hat), which is smallest (most
                # ill-conditioned) at high t -- without this, a prediction error there gets
                # amplified and compounds multiplicatively over ~1000 sequential steps,
                # saturating the output instead of converging to an image.
                x0_pred = ((x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)).clamp(-1, 1)
                predicted_noise = (x - torch.sqrt(alpha_hat) * x0_pred) / torch.sqrt(1 - alpha_hat)

                if i > 1:
                    noise = torch.randn(x.shape, device=self.device, generator=generator)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


if __name__ == "__main__":
    shape = (3, 64, 64)
    batch = 2
    model = UNet(input_shape=shape, output_shape=shape, num_attributes=10)
    print("Num params:", sum(p.numel() for p in model.parameters()))
    x = torch.rand(batch, *shape)
    t = torch.randint(0, 1000, (batch,)).long()
    attrs = torch.randint(0, 2, (batch, 10)).float()
    print("output:", model(x, t, attrs).shape)
