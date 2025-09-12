# -*- coding: utf-8 -*-
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_Windows*B, window_size:Mh, window_size:Mw, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # view: [B, H, W, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]

    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    return windows


def window_partition2(x, window_size: tuple):
    """ Split the feature map to windows.
            [B, C, H, W] --> [num_Windows*B, Mh*Mw, C]
            num_Windows = H // Mh * W // Mw

    Args:
        x: (B, C, H, W)
        window_size (tuple[int]): window size

    Returns:
        windows: (num_windows*B, window_size * window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # view: [B, C, H, W] -> [B, C, H//Mh, Mh, W//Mw, Mw]

    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0] * window_size[1], C)
    # permute: [B, C, H//Mh, Mh, W//Mw, Mw] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh*Mw, C]
    return windows


def window_reverse(windows, window_size: int, H: int, W: int, C: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    return x


def window_reverse2(windows, window_size: tuple, H: int, W: int, C: int):
    """ Windows reverse to feature map.
            [num_Windows*B, Mh*Mw, C] --> [B, C, H, W]
            num_Windows = H // Mh * W // Mw
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size([Mh, Mw])
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]

    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    return x


class ConvEmbed(nn.Module):
    """ Z-spectrum to Conv Stem Embedding

    Args:
        patch_size (int): Patch token size. Default: 11.
        in_offs (int): Number of input Z-spectrum offsets. Default: 41.
        out_offs (int): Number of output Z-spectrum offsets. Default: 32*32.
        embed_dim (int): Number of linear projection output channels.
            Default: 96.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: None
    """

    def __init__(self, patch_size=11, in_offs=41, out_offs=32 * 32, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_offs
        self.embed_dim = embed_dim
        patches_resolution = [int(math.sqrt(out_offs)), int(math.sqrt(out_offs))]
        self.patches_resolution = patches_resolution
        self.num_patches = out_offs
        self.proj = nn.Sequential(
            nn.Conv2d(in_offs, in_offs, kernel_size=3, stride=1, padding=1),  # input[41, 5, 5]  output[41, 5, 5]
            nn.BatchNorm2d(in_offs),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_offs, out_offs, kernel_size=1, stride=1, padding=0),  # output[1024, 5, 5]
            nn.BatchNorm2d(out_offs),
            nn.ReLU(inplace=True)
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.2),  # 使神经元有 p 的几率失活
            nn.Linear(patch_size[0] * patch_size[1], embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            # nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, embed_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        # proj: [B, C, patch_H, patch_W] -> [B, offset_num, patch_H, patch_W]

        x = x.flatten(2)
        # flatten: [B, offset_num, patch_H, patch_W] -> [B, offset_num, patch_H*patch_W]

        x = self.fc_layer(x)
        # fc_layer: [B, offset_num, patch_H*patch_W] -> [B, offset_num, embed_dim]

        _, patch_num, embed_dim = x.shape
        x = x.permute(0, 2, 1)
        # transpose: [B, offset_num, embed_dim] -> [B, embed_dim, offset_num]

        # x = self.norm(x)
        x = x.reshape([-1, self.embed_dim, self.patches_resolution[0], self.patches_resolution[0]])
        # reshape: [B, embed_dim, offset_num] -> [B, embed_dim, Wh, Ww]
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ConvMerging(nn.Module):
    r""" Conv Merging Layer.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Output channels after the merging layer.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        # 使用卷积层进行下采样
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # 使用池化层进行下采样
        # self.reduction = nn.Sequential(
        #     nn.Conv2d(dim, out_dim, kernel_size=1, stride=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape([B, H, W, C]).permute(0, 3, 1, 2)
        # reshape: [B, Wh*Ww, C] -> [B, Wh, Ww, C]
        # transpose: [B, Wh, Ww, C] -> [B, C, Wh, Ww]

        x = self.norm(x)
        # B, C, H, W -> B, H*W, C
        x = self.reduction(x).flatten(2).permute(0, 2, 1)
        # reduction: [B, C, Wh, Ww] -> [B, C_out(2C), Wh/2, Ww/2]
        # flatten: [B, C_out(2C), Wh/2, Ww/2] -> [B, C_out(2C), Wh/2*Ww/2]
        # transpose: [B, C_out(2C), Wh/2*Ww/2] -> [B, Wh/2*Ww/2, C_out(2C)]
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 如果out_features、hidden_features有传入，则为传入的值，如果没有传入，就保持与in_features相同
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixingAttention(nn.Module):
    r""" Mixing Attention Module.
    Modified from Window based multi-head self attention (W-MSA) module
    with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        val2: val channel interaction and spatial_interaction. Default: False
    """

    def __init__(self,
                 dim,
                 window_size,
                 dwconv_kernel_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 val1=False,
                 val2=False):
        super().__init__()
        self.val1 = val1
        self.val2 = val2
        self.dim = dim
        attn_dim = dim // 2
        self.window_size = window_size  # [Wh, Ww]
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # self.add_parameter("relative_position_bias_table", self.relative_position_bias_table)

        # get pair-wise relative position index for each token
        # inside the window
        relative_coords = self._get_rel_pos()
        self.relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        # self.register_buffer("relative_position_index", self.relative_position_index)

        # prev proj layer
        self.proj_attn = nn.Linear(dim, dim // 2)
        self.proj_attn_norm = nn.LayerNorm(dim // 2)
        self.proj_cnn = nn.Linear(dim, dim)
        self.proj_cnn_norm = nn.LayerNorm(dim)

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=self.dwconv_kernel_size,
                padding=self.dwconv_kernel_size // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),
        )
        self.projection = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm2d(dim // 2)

        # window-attention branch
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)  # 从输入a经过Wq Wk Wv得到q k v
        self.qkv_val = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 从输入a经过Wq Wk Wv得到q k v
        self.attn_drop = nn.Dropout(attn_drop)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        self.attn_norm = nn.LayerNorm(dim // 2)

        # final projection
        self.proj = nn.Linear(dim, dim)  # 多头注意力的Wo的线性映射，使用全连接层实现
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos(self):
        """ Get pair-wise relative position index for each token inside the window.

        Args:
            window_size (tuple[int]): window size
        """
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        # 2, Wh, Ww
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        return relative_coords

    def forward(self, x, H, W, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
                or None
        """
        # B * H // win * W // win x win*win x C
        if not self.val1:
            x_atten = self.proj_attn_norm(self.proj_attn(x))
        else:
            x_atten = x
        if not self.val1:
            x_cnn = self.proj_cnn_norm(self.proj_cnn(x))
            # B * H // win * W // win x win*win x C --> B, C, H, W
            x_cnn = window_reverse2(x_cnn, self.window_size, H, W, x_cnn.shape[-1])

            # conv branch
            x_cnn = self.dwconv3x3(x_cnn)
            channel_interaction = self.channel_interaction(F.adaptive_avg_pool2d(x_cnn, output_size=1))
            x_cnn = self.projection(x_cnn)

        # attention branch
        B_, N, C = x_atten.shape
        if not self.val1:
            qkv = self.qkv(x_atten).reshape(
                [B_, N, 3, self.num_heads, C // self.num_heads]).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv_val(x_atten).reshape(
                [B_, N, 3, self.num_heads, C // self.num_heads]).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # channel interaction
        if not self.val1:
            x_cnn2v = F.sigmoid(channel_interaction).reshape([-1, 1, self.num_heads, 1, C // self.num_heads])
            v = v.reshape([x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads])
        if not (self.val2 or self.val1):   # todo
            v = v * x_cnn2v
        v = v.reshape([-1, self.num_heads, N, C // self.num_heads])

        # Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        index = self.relative_position_index.view(-1)
        relative_position_bias = self.relative_position_bias_table[index].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # [Wh*Ww, Wh*Ww, nH]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_atten = (attn @ v).permute(0, 2, 1, 3).reshape(B_, N, C)
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]

        # spatial interaction
        if not self.val1:   # todo
            x_spatial = window_reverse2(x_atten, self.window_size, H, W, C)
            # window_reverse2: [B*nW, Mh*Mw, C//2] -> [B, C//2, Hp, Wp]
            spatial_interaction = self.spatial_interaction(x_spatial)
            # spatial_interaction: [B*nW, Mh*Mw, C//2] -> [B, 1, Hp, Wp]
        if not (self.val2 or self.val1):  # todo
            x_cnn = F.sigmoid(spatial_interaction) * x_cnn
            # x_cnn: [B, C//2, Hp, Wp]
        if not self.val1:
            x_cnn = self.conv_norm(x_cnn)
            # B, C, H, W --> B * H // win * W // win x win*win x C
            x_cnn = window_partition2(x_cnn, self.window_size)

        # concat
        if not self.val1:   # todo
            x_atten = self.attn_norm(x_atten)
            x = torch.concat([x_atten, x_cnn], dim=-1)
            # concat: [B*nW, Mh*Mw, C]
        else:
            x = x_atten

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixingBlock(nn.Module):
    r""" Mixing Block in MixFormer.
    Modified from Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        shift_size (int): Shift size for SW-MSA.
            We do not use shift in MixFormer. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 dwconv_kernel_size=3,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 val1=False,
                 val2=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert self.shift_size == 0, "No shift in MixFormer"

        self.norm1 = norm_layer(dim)
        self.attn = MixingAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            val1=val1,
            val2=val2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        # shortcut: [B, Wh*Ww, C]
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])
        # x: [B, Wh*Ww, C] -> [B, Wh, Ww, C]

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # x = F.pad(x, [0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t])
        _, Hp, Wp, _ = x.shape
        # x: [B, Wh, Ww, C] -> [B, Hp, Wp, C]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, window_size*window_size, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, Hp, Wp, mask=attn_mask)  # [nW*B, window_size*window_size, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, C)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x