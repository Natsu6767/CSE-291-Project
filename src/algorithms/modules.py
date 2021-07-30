import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_out_shape_cuda(in_shape, layers):
    x = torch.randn(*in_shape).cuda().unsqueeze(0)
    return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def vit_orthogonal_init(m, n='', head_bias=0.):
    """Custom weight init for ViT layers"""
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class NormalizeImg(nn.Module):
    def __init__(self, mean_zero=False):
        super().__init__()
        self.mean_zero = mean_zero

    def forward(self, x):
        if self.mean_zero:
            return x / 255. - 0.5
        return x / 255.


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def __init__(self, obs_shape=None, out_dim=None):
        super().__init__()
        self.out_shape = obs_shape
        self.out_dim = out_dim

    def forward(self, x):
        return x


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class StateMlp(nn.Module):
    def __init__(self, in_shape, hidden_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape[0], hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh()
        )
        self.apply(orthogonal_init)
        self.out_shape = (out_dim,)

    def forward(self, x):
        return self.projection(x)


class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape[0], out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh()
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.projection(x)


class SharedTransformer(nn.Module):
    def __init__(self, obs_shape, num_layers=4, num_heads=8, embed_dim=128, patch_size=8, mlp_ratio=1., qvk_bias=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.frame_stack = obs_shape[0] // 3
        self.img_size = obs_shape[-1]
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qvk_bias = qvk_bias

        self.preprocess = nn.Sequential(NormalizeImg())
        self.transformer = VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=self.frame_stack * 3,
            embed_dim=embed_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qvk_bias,
        ).cuda()
        self.out_shape = _get_out_shape_cuda(obs_shape, nn.Sequential(self.preprocess, self.transformer))

    def forward(self, x):
        x = self.preprocess(x)
        return self.transformer(x)


class TransformerMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TransformerMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=84, patch_size=14, in_chans=3, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]
        assert self.num_patches == 144, f'unexpected number of patches: {self.num_patches}'

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=96, patch_size=8, in_chans=9, embed_dim=128, depth=4,
                 num_heads=8, mlp_ratio=1., qkv_bias=True, qk_scale=None):
        super().__init__()
        self.embed_dim = embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, act_layer=nn.GELU)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(vit_orthogonal_init)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32, project=False, mean_zero=False):
        super().__init__()
        #assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.project = project

        #self.layers = [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        #self.layers = [NormalizeImg(mean_zero), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        #self.layers = [nn.Conv2d(obs_shape[0], 32, kernel_size=1, stride=1, padding=0)]
        """for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))"""
        #self.layers = nn.Sequential(*self.layers)
        self.out_shape = obs_shape# _get_out_shape(obs_shape, self.layers)
        #self.apply(orthogonal_init)

    def forward(self, x):
        if self.project:
            x = x.view(-1, x.size(1) * x.size(2), x.size(3), x.size(4))
        return x#self.layers(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)


class Integrator(nn.Module):
    def __init__(self, in_shape_1, in_shape_2, num_filters=32):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_shape_1[0] + in_shape_2[0], num_filters, (1, 1))
        self.apply(weight_init)

    def forward(self, x):
        x = self.conv1(self.relu(x))
        return x


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)


class MultiViewEncoder(nn.Module):
    def __init__(self, shared_cnn_1, shared_cnn_2, integrator, head_cnn, projection):
        super().__init__()
        self.shared_cnn_1 = shared_cnn_1
        self.shared_cnn_2 = shared_cnn_2
        self.integrator = integrator
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x1, x2, detach=False):
        x1 = self.shared_cnn_1(x1)
        x2 = self.shared_cnn_1(x2)

        # Concatenate features along channel dimension
        x = torch.cat((x1, x2), dim=1)  # 1, 64, 21, 21
        x = self.integrator(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)


class EfficientActor(nn.Module):
    def __init__(self, out_dim, projection_dim, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.layers = nn.Sequential(
            nn.Linear(out_dim, projection_dim),
            nn.LayerNorm(projection_dim), nn.Tanh(),
            nn.Linear(projection_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.apply(orthogonal_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.layers(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class Actor(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max, multiview=False):
        super().__init__()
        self.encoder = encoder
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.mlp.apply(orthogonal_init)
        self.multiview = multiview

    def forward(self, x_in, compute_pi=True, compute_log_pi=True, detach=False):
        if self.multiview:
            x1, x2 = x_in[:, :3, :, :], x_in[:, 3:6, :, :]
            x = self.encoder(x1, x2, detach)
        else:
            x = self.encoder(x_in, detach)

        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(orthogonal_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        return self.trunk(torch.cat([obs, action], dim=1))


class EfficientCritic(nn.Module):
    def __init__(self, out_dim, projection_dim, action_shape, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(out_dim, projection_dim),
                                        nn.LayerNorm(projection_dim), nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(projection_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(projection_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.apply(orthogonal_init)

    def forward(self, obs, action):
        h = torch.cat([self.projection(obs), action], dim=-1)
        return self.Q1(h), self.Q2(h)


class Critic(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, multiview=False):
        super().__init__()
        self.encoder = encoder
        self.Q1 = QFunction(
            self.encoder.out_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.out_dim, action_shape[0], hidden_dim
        )
        self.multiview = multiview

    def forward(self, x_in, action, detach=False):
        if self.multiview:
            x1, x2 = x_in[:, :3, :, :], x_in[:, 3:6, :, :]
            x = self.encoder(x1, x2, detach)
        else:
            x = self.encoder(x_in, detach)

        return self.Q1(x, action), self.Q2(x, action)
