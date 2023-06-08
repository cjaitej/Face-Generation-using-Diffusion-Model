import torch
import torch.nn as nn
import math

device = torch.device('cuda')

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block, self).__init__()
        time_emb_dim = 32
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.time_mlp =  nn.Linear(time_emb_dim, out_c)
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)
        self.activation = nn.ReLU()

    def forward(self, x, time_emb):
        x = y = self.conv1(x)
        x = self.norm1(self.conv2(x))
        x = self.activation(x)
        x = self.norm2(self.conv3(x)) + y
        x = self.activation(x)
        time_emb = self.activation(self.time_mlp(time_emb))
        x = x + time_emb[(..., ) + (None, ) * 2]
        return x


class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(Encoder, self).__init__()
        self.conv_block = conv_block(in_c, out_c)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv_block(x, t)
        p = self.pool(x)
        return p, x


class Decoder(nn.Module):
    def __init__(self, in_c, out_c, mode=''):
        super(Decoder, self).__init__()
        self.transpose_layer = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv_block = conv_block(out_c + out_c, out_c)
        if mode == 'final_layer':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

    def forward(self, input, skip, t):
        x = self.transpose_layer(input)
        x = torch.cat([x, skip], axis=1)
        x = self.conv_block(x, t)
        x = self.activation(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_features, num_heads=1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=num_heads)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=1)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=1)
        self.conv3 = nn.Conv2d(n_features, n_features, kernel_size=1)

    def forward(self, input):
        output = torch.zeros_like(input)
        for n, i in enumerate(input):
            query = self.conv1(i)
            key = self.conv2(i)
            value = self.conv3(i)

            output[n] = self.attention(query, key, value, need_weights=False)[0]

        return output


class UNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(UNet, self).__init__()
        time_emb_dim = 32

        self.channels = [32, 64, 128, 256, 512]

        self.dim = [input_shape[-1]//(2**i) for i in range(1, len(self.channels) + 1)]

        self.encoder_list = nn.ModuleList([Encoder(input_shape[0], 32)] + [Encoder(in_c, in_c*2) for in_c in self.channels[:-1]])

        self.attention_list = nn.ModuleList([Attention(dim, n_features, 1) for dim, n_features in zip(self.dim, self.channels)])

        self.decoder_list = nn.ModuleList([Decoder(in_c*2, in_c) for in_c in self.channels[::-1][:-1]] + [Decoder(64, 32, mode='final_layer')])

        self.bottle_neck = conv_block(512, 1024)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.output = nn.Conv2d(32, output_shape[0], kernel_size = 1, padding=0)
        # self.final_activation = nn.Sigmoid()

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        intermediate_values = []
        for encoder, attention in zip(self.encoder_list, self.attention_list):
            x, i = encoder(x, t)
            x = attention(x)
            intermediate_values.append(i)

        x = self.bottle_neck(x, t)

        for decoder, skip in zip(self.decoder_list, intermediate_values[::-1]):
            x = decoder(x, skip, t)

        x = self.output(x)

        return x

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if i%100 == 0:
                    print("=", end="")
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x



if __name__ == "__main__":
    shape = (3, 256, 256)
    batch = 5
    # model = DiffusionModel(10)
    model = UNet(input_shape=shape, output_shape=shape)
    # input = torch.rand(10, 3, 512, 512)
    # print("Num params: ", sum(p.numel() for p in model.parameters()))
    # model = Attention(256, 32, 1)
    input = torch.rand(batch, 3, 256, 256)
    t = torch.randint(0, 10, (batch,)).long()
    pred = model(input, t)
    print(pred[0].shape, pred[1].shape)
