import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

if torch.cuda.is_available():
    dev = "cuda:1"
else:
    dev = "cpu"

device = torch.device(dev)


# print("Device:", device)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
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


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


# class CustomAct(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu

    def forward(self, x):
        return self.act_layer(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    #         self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))
    #         self.noise_strength_2 = torch.nn.Parameter(torch.zeros([]))
    def forward(self, x):
        #         x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        x = self.fc1(x)
        x = self.act(x)  ###################gelu or leakey
        x = self.drop(x)  ###############类似于正则化 很像 batch normalization
        #         x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_2
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, que_dim, key_dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.que_dim = que_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        head_dim = que_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        #         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_transform = nn.Linear(que_dim, que_dim, bias=qkv_bias)
        self.k_transform = nn.Linear(key_dim, que_dim, bias=qkv_bias)
        self.v_transform = nn.Linear(key_dim, que_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(que_dim, que_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()

        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x, embedding):
        B, N, C = x.shape
        B, E_N, E_C = embedding.shape

        # transform
        q = self.q_transform(x)
        k = self.k_transform(embedding)
        v = self.v_transform(embedding)
        # reshape
        q = q.reshape(B, N, self.num_heads, self.que_dim // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N, C)
        k = k.reshape(B, E_N, self.num_heads, self.que_dim // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N, C)
        v = v.reshape(B, E_N, self.num_heads, self.que_dim // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N, C)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.size(-1) == v.size(-2), f"attn.size: {attn.size()}, v.size:{v.size()}"
        output = self.mat(attn, v).transpose(1, 2).reshape(B, N, self.que_dim)
        output = self.proj(output)
        output = self.proj_drop(output)
        return x + output


class MHA(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MHA, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self.scores = None

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q, k, v = (self.split(x, (self.nhead, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.dropout(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge(h, 2)
        self.scores = scores
        return h, scores

    def split(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)

    def merge(self, x, n_dims):
        s = x.size()
        assert 1 < n_dims < len(s)
        return x.view(*s[:-n_dims], -1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)


class Block(nn.Module):
    def __init__(self, dim, embedding_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.window_size = window_size
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cross_attention = CrossAttention(que_dim=dim, key_dim=embedding_dim, num_heads=num_heads)

    def forward(self, inputs):
        x, embedding = inputs
        x = self.cross_attention(x, embedding)  ##############
        B, N, C = x.size()
        H = W = int(np.sqrt(N))
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, N, C)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return [x, embedding]


class StageBlock(nn.Module):
    def __init__(self, depth, dim, embedding_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.depth = depth
        models = [Block(
            dim=dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=window_size
        ) for i in range(depth)]
        self.block = nn.Sequential(*models)

    def forward(self, x, embedding):
        #         for blk in self.block:
        #             # x = blk(x)
        #             checkpoint.checkpoint(blk, x)
        #         x = checkpoint.checkpoint(self.block, x)
        x = self.block([x, embedding])
        return x


def pixel_upsample(x, H, W):
    # B, N, C -> B, H*W, C
    # H: H*2, W:W*2
    # print(H, W)
    B, N, C = x.size()  # n, 64, 224
    assert N == H * W
    x = x.permute(0, 2, 1)  # n, 224, 64
    x = x.view(-1, C, H, W)  # n, 224, 8, 8
    x = nn.PixelShuffle(2)(x)  # n, 224/4, 16, 16
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)  # n, 224/4, 16*16
    x = x.permute(0, 2, 1)  # n, 16*16, 224/4
    # print(x.shape)
    return x, H, W


def bicubic_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def updown(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=4, mode='bicubic')
    x = nn.AvgPool2d(4)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class TGenerator(nn.Module):
    def __init__(self, args, device1_1="cuda:1", device1_2="cuda:1", device1_3="cuda:1", img_size=128,
                 patch_size=16, in_chans=3, num_classes=10, embed_dim=128, depth=5,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super(TGenerator, self).__init__()
        self.device1_1 = device1_1
        self.device1_2 = device1_2
        self.device1_3 = device1_3
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        self.window_size = args.g_window_size
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        num_heads = args.num_heads
        self.l2_size = 0

        if self.l2_size == 0:
            self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        elif self.l2_size > 1000:
            self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.l2_size // 16)
            self.l2 = nn.Sequential(
                nn.Linear(self.l2_size // 16, self.l2_size),
                nn.Linear(self.l2_size, self.embed_dim)
            )
        else:
            self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.l2_size)
            self.l2 = nn.Linear(self.l2_size, self.embed_dim)

        self.l1 = self.l1  # .to(device1_1)
        if self.l2_size != 0:
            self.l2 = self.l2  # .to(device1_1)
        self.embedding_transform = nn.Linear(args.latent_dim,
                                             (self.bottom_width ** 2) * self.embed_dim)  # .to(device1_1)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim))  # .to(device1_1)
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, embed_dim))  # .to(device1_1)
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim))  # .to(device1_2)
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width * 8) ** 2, embed_dim // 4))  # .to(device1_2)
        self.pos_embed_5 = nn.Parameter(
            torch.zeros(1, (self.bottom_width * 16) ** 2, embed_dim // 16))  # .to(device1_3)
        self.pos_embed_6 = nn.Parameter(
            torch.zeros(1, (self.bottom_width * 32) ** 2, embed_dim // 64))  # .to(device1_3)

        self.embed_pos = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim))

        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4,
            self.pos_embed_5,
            self.pos_embed_6
        ]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # stochastic depth decay rule
        self.blocks_1 = StageBlock(
            depth=depth[0],
            dim=embed_dim,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=8
        )  # .to(device1_1)
        self.blocks_2 = StageBlock(
            depth=depth[1],
            dim=embed_dim,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=16
        )  # .to(device1_1)
        self.blocks_3 = StageBlock(
            depth=depth[2],
            dim=embed_dim,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=32
        )  # .to(device1_2)
        self.blocks_4 = StageBlock(
            depth=depth[3],
            dim=embed_dim // 4,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size
        )  # .to(device1_2)
        self.blocks_5 = StageBlock(
            depth=depth[4],
            dim=embed_dim // 16,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size
        )  # .to(device1_3)
        # self.blocks_6 = StageBlock(
        #     depth=depth[5],
        #     dim=embed_dim // 64,
        #     embedding_dim=embed_dim,
        #     num_heads=num_heads,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=0,
        #     act_layer=act_layer,
        #     norm_layer=norm_layer,
        #     window_size=self.window_size
        # ).to(device1_3)

        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim // 16, 3, 1, 1, 0)
        )  # .to(device1_3)

    def forward(self, z):
        if self.args.latent_norm:
            latent_size = z.size(-1)
            z = (z / z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
        if self.l2_size == 0:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
            # print(x.shape)
        elif self.l2_size > 1000:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size // 16)
            x = self.l2(x)
        else:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size)
            x = self.l2(x)

        # input noise
        # x = x + self.pos_embed[0].to(x.get_device())
        # B = x.size(0)
        # H, W = self.bottom_width, self.bottom_width

        # embedding
        embedding = self.embedding_transform(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        embedding = embedding + self.embed_pos.to(embedding.get_device())  # n, 64, 224

        # print(x.shape)
        x = x + self.pos_embed[0].to(x.get_device())
        B = x.size()
        H, W = self.bottom_width, self.bottom_width
        x, _ = self.blocks_1(x, embedding)  # n, 64, 224
        # print(x.shape)
        # print(x.shape)

        x, H, W = bicubic_upsample(x, H, W)
        x = x + self.pos_embed[1].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_2(x, embedding)  # n, 256, 224

        # print(x.shape)
        # print(x.shape)
        x = x  # .to(self.device1_2)
        embedding = embedding  # .to(self.device1_2)

        x, H, W = bicubic_upsample(x, H, W)
        x = x + self.pos_embed[2].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_3(x, embedding)  # n. 1024, 224
        # print(x.shape)

        x, H, W = pixel_upsample(x, H, W)
        # print(x.shape)
        x = x + self.pos_embed[3].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_4(x, embedding)  # n, 4096, 56
        # print(x.shape)
        x = x  # .to(self.device1_3)
        embedding = embedding  # .to(self.device1_3)

        x, H, W = updown(x, H, W)
        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[4].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_5(x, embedding)  # n, 16384, 14
        # print(x.shape)

        # x, H, W = pixel_upsample(x, H, W)
        # # print(x.shape, self.pos_embed[5].shape)
        # x = x + self.pos_embed[5].to(x.get_device())
        # B, _, C = x.size()
        # x, _ = self.blocks_6(x, embedding)
        # print(x.shape)

        x = x.permute(0, 2, 1).view(B, C, 128, 128)
        output = self.deconv(x)

        return output


class PositionEmbedding(nn.Module):
    def __init__(self, input_seq, d_model):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, input_seq, d_model))

    def forward(self, x):
        x = x + self.position_embedding
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.ff1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.ff2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        x = self.ff2(F.gelu(self.ff1(x)))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.attn = MHA(d_model=d_model, nhead=nhead, dropout=dropout)
        self.linproj = nn.Linear(in_features=d_model, out_features=d_model)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, scores = self.attn(self.norm1(x))
        h = self.dropout(self.linproj(h))
        x = x + h
        h = self.dropout(self.ff(self.norm2(x)))
        x = x + h
        return x, scores


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x):
        scores = []
        for block in self.blocks:
            x, score = block(x)
            scores.append(score)
        return x, scores


class TDiscriminator(nn.Module):
    def __init__(self,
                 patches=(8, 8),  # Patch size: height width
                 d_model=128,  # Token Dim
                 d_ff=128,  # Feed Forward Dim
                 num_heads=4,  # Num MHA
                 num_layers=3,  # Num Transformer Layers
                 dropout=.1,  # Dropout rate
                 image_size=(1, 128, 128),  # channels, height, width
                 num_classes=1,  # Dataset Categories
                 ):
        super(TDiscriminator, self).__init__()

        self.image_size = image_size

        # ---- 1 Patch Embedding ---
        c, h, w = image_size  # image sizes

        ph, pw = patches  # patch sizes

        n, m = h // ph, w // pw
        seq_len = n * m  # number of patches

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels=c, out_channels=d_model, kernel_size=(ph, pw), stride=(ph, pw))

        # Class token
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embedding
        self.position_embedding = PositionEmbedding(input_seq=(seq_len + 1), d_model=d_model)

        # Transformer
        self.transformer = TransformerEncoder(num_layers=num_layers, d_model=d_model, nhead=num_heads,
                                              d_ff=d_ff, dropout=dropout)

        # Classifier head
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp = nn.Linear(in_features=d_model, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, ph, pw = x.shape

        x = self.patch_embedding(x)  # n, 128, 28, 28
        x = x.flatten(2).transpose(1, 2)  # n, 784, 128

        x = torch.cat((self.class_embedding.expand(b, -1, -1), x), dim=1)  # n, 785, 128

        x = self.position_embedding(x)

        x, scores = self.transformer(x)
        x = self.norm(x)[:, 0]
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

    def represent(self, x):
        b, c, ph, pw = x.shape

        x = self.patch_embedding(x)  # n, 128, 28, 28
        x = x.flatten(2).transpose(1, 2)  # n, 784, 128

        x = torch.cat((self.class_embedding.expand(b, -1, -1), x), dim=1)  # n, 785, 128

        x = self.position_embedding(x)

        x, scores = self.transformer(x)
        x = self.norm(x)[:, 0]
        return x


class VGG(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channel, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.max_pool = nn.MaxPool2d(2, 2, 0)
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        return x


class CNNGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape=(3, 128, 128)):
        super().__init__()
        self.label_emb = nn.Embedding(latent_dim, latent_dim)
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, condition):
        # Concatenate label embedding and image to produce input
        # z = torch.rand((condition.shape[0], self.latent_dim)).to(condition.get_device())
        # condition = torch.cat((z, condition), dim=1)
        img = self.model(condition)
        img = img.view(img.size(0), *self.img_shape)
        return img


class LinearBasicBlock(nn.Module):
    def __init__(self, in_feat, out_feat, number, normalize=True):
        super().__init__()
        self.layers = []
        for i in range(number):
            layer = [nn.Linear(in_feat, in_feat)]
            if normalize:
                layer.append(nn.BatchNorm1d(in_feat, 0.8))
            layer.append(nn.LeakyReLU(0.2, inplace=True))
            self.layers.append(nn.Sequential(*layer))
        layer = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layer.append(nn.BatchNorm1d(out_feat, 0.8))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        self.final = nn.Sequential(*layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        x = self.final(x)
        return x


class ResidualGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape=(3, 128, 128), layers=[3, 4, 6, 3]):
        super().__init__()
        self.label_emb = nn.Embedding(latent_dim, latent_dim)
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            LinearBasicBlock(latent_dim, 128, layers[0], normalize=False),
            LinearBasicBlock(128, 256, layers[1]),
            LinearBasicBlock(256, 512, layers[2]),
            LinearBasicBlock(512, 1024, layers[3]),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, condition):
        # Concatenate label embedding and image to produce input
        # z = torch.rand((condition.shape[0], self.latent_dim)).to(condition.get_device())
        # condition = torch.cat((z, condition), dim=1)
        img = self.model(condition)
        img = img.view(img.size(0), *self.img_shape)
        return img


class CNN(nn.Module):
    def __init__(self, ndf, nc, num_classes, diffaugment='color,translation,cutout'):
        super(CNN, self).__init__()
        self.diffaugment = diffaugment
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, num_classes, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # self.output = nn.Linear(self.flatten_shape, 1)

    def forward(self, img, do_augment=True):
        # if do_augment:
        #     img = DiffAugment(img, policy=self.diffaugment)
        out = self.main(img)
        # print(out.shape)
        # print(out.shape)
        # print(self.flatten_shape)
        # out = out.view(-1, self.flatten_shape)
        # out = self.output(out)
        out = out.squeeze()

        # out = out * 2 - 1
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, upsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        if upsample is not None:
            self.conv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,
                                            kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False)
        '''
        kernel_size=3,padding=1,stride=1时,output=(input-3+2*1)/1+1=input
        kernel_size=3,padding=1,stride=2时,output=(input-3+2*1)/2+1=input/2+0.5(向下取整)=input/2
        '''
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.upsample = upsample

    def forward(self, x):
        identity = x
        if self.upsample is not None:
            identity = self.upsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.shape)

        out += identity
        out = self.relu(out)

        return out


class ResidualGeneratorWithTConv(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 groups=1,
                 latent_dim=512,
                 width_per_group=64):
        super(ResidualGeneratorWithTConv, self).__init__()
        self.in_channel = 64
        self.in_channel_origin = self.in_channel

        self.groups = groups
        self.width_per_group = width_per_group
        self.reshape = nn.Linear(latent_dim, 4 * 4 * self.in_channel)  # 4*4
        self.conv1 = nn.ConvTranspose2d(self.in_channel, 64, 4, stride=2, padding=1)  # 8*8
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2,
        #                        padding=3, bias=False)  # RGB三通道图像in_channels=3,conv1的输出层等于后续的输入层
        # output=(input-7+2x3)/2+1=input/2=224/2=112
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 8*8
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=2)  # conv2_x
        self.layer2 = self._make_layer(block, 32, blocks_num[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 16, blocks_num[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 8, blocks_num[3], stride=2)  # conv5_x
        self.out = nn.Conv2d(self.in_channel, 3, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        upsample = None
        if stride != 1 or self.in_channel != channel // block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channel, channel // block.expansion, 4, stride=2, padding=1, bias=False),
                # nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            upsample=upsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel // block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.reshape(x)
        x = x.view(x.shape[0], self.in_channel_origin, 4, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 8*8

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out(x)

        return x


def resnetgen34():
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResidualGeneratorWithTConv(BasicBlock, [3, 4, 6, 3])


class CrossDiscriminator(nn.Module):
    def __init__(self,
                 patches=(8, 8),  # Patch size: height width
                 d_model=128,  # Token Dim
                 d_ff=128,  # Feed Forward Dim
                 num_heads=4,  # Num MHA
                 num_layers=4,  # Num Transformer Layers
                 dropout=.1,  # Dropout rate
                 image_size=(3, 128, 128),  # channels, height, width
                 num_classes=1,  # Dataset Categories
                 ):
        super(CrossDiscriminator, self).__init__()

        self.image_size = image_size

        # ---- 1 Patch Embedding ---
        c, h, w = image_size  # image sizes

        ph, pw = patches  # patch sizes

        n, m = h // ph, w // pw
        seq_len = n * m  # number of patches

        # Patch embedding
        self.patch_embedding1 = nn.Conv2d(in_channels=c, out_channels=d_model, kernel_size=(ph, pw),
                                          stride=(ph, pw))  # n, d, h/ph, w/pw
        self.patch_embedding2 = nn.Conv2d(in_channels=c, out_channels=d_model, kernel_size=(ph, pw),
                                          stride=(ph, pw))

        # Class token
        self.class_embedding1 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.class_embedding2 = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embedding
        self.position_embedding1 = PositionEmbedding(input_seq=(seq_len + 1), d_model=d_model)
        self.position_embedding2 = PositionEmbedding(input_seq=(seq_len + 1), d_model=d_model)

        # Transformer
        self.transformer1 = TransformerEncoder(num_layers=num_layers, d_model=d_model, nhead=num_heads,
                                              d_ff=d_ff, dropout=dropout)
        self.transformer2 = TransformerEncoder(num_layers=num_layers, d_model=d_model, nhead=num_heads,
                                               d_ff=d_ff, dropout=dropout)
        self.cross = CrossAttention(d_model, d_model, num_heads=num_heads)

        # Classifier head
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp1 = nn.Linear(in_features=d_model, out_features=num_classes)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp2 = nn.Linear(in_features=d_model, out_features=num_classes)

    def forward(self, x1, x2):
        b, c, ph, pw = x1.shape

        x1 = self.patch_embedding1(x1)  # n, 128, 16, 16 (n, d, h/ph, w/pw)
        x2 = self.patch_embedding2(x2)
        x1 = x1.flatten(2).transpose(1, 2)  # n, 16*16, 128 (n, hw/(phpw), d)
        x2 = x2.flatten(2).transpose(1, 2)

        #  embed: (n, 1, d)
        x1 = torch.cat((self.class_embedding1.expand(b, -1, -1), x1), dim=1)  # n, hw/(phpw)+1, d
        x2 = torch.cat((self.class_embedding2.expand(b, -1, -1), x2), dim=1)
        # print(x.shape)

        x1 = self.position_embedding1(x1)  # n, hw/(phpw)+1, d
        x2 = self.position_embedding2(x2)
        # print(x1.shape)
        x1, _ = self.transformer1(x1)
        x2, _ = self.transformer2(x2)

        x1_pre = self.cross(x1, x2)
        x1_pre = self.norm1(x1_pre)[:, 0]
        x1_pre = self.mlp1(x1_pre)
        x2_pre = self.cross(x2, x1)
        x2_pre = self.norm2(x2_pre)[:, 0]
        x2_pre = self.mlp2(x2_pre)  # drop sigmoid
        # print((torch.exp(x1_pre[:, 0]) / (torch.exp(x1_pre[:, 0]) + torch.exp(x2_pre[:, 0]))).shape)
        # print(F.sigmoid(x1_pre[:, 1:]).shape)
        return torch.concat(
            [(torch.exp(x1_pre[:, 0]) / (torch.exp(x1_pre[:, 0]) + torch.exp(x2_pre[:, 0]))).unsqueeze(1),
             F.sigmoid(x1_pre[:, 1:])],
            dim=1)
