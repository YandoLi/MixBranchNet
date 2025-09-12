import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MixingBlock import MixingBlock


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class ConvEmbed(nn.Module):
    """ Z-spectrum to Conv Stem Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_c (int): Number of input Z-spectrum offsets. Default: 41.
        embed_dim (int): Number of linear projection output channels.
            Default: 96.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: None
    """

    def __init__(self, patch_size=11, in_offs=41, embed_dim=96, norm_layer=None, out_offs=48):
        super().__init__()
        # patch_size = (patch_size, patch_size)
        # self.patch_size = patch_size
        # self.in_chans = in_offs
        # self.embed_dim = embed_dim
        # patches_resolution = [int(math.sqrt(out_offs)), int(math.sqrt(out_offs))]
        # self.patches_resolution = patches_resolution
        # self.num_patches = out_offs
        # self.proj = nn.Sequential(
        #     nn.Conv2d(in_offs, in_offs, kernel_size=3, stride=1, padding=1),  # input[41, 5, 5]  output[41, 5, 5]
        #     nn.BatchNorm2d(in_offs),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_offs, out_offs, kernel_size=1, stride=1, padding=0),  # output[1024, 5, 5]
        #     nn.BatchNorm2d(out_offs),
        #     nn.ReLU(inplace=True)
        # )
        # self.fc_layer = nn.Sequential(
        #     nn.Dropout(p=0.2),  # 使神经元有 p 的几率失活
        #     nn.Linear(patch_size[0] * patch_size[1], embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     # nn.Linear(1024, 1024),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(1024, embed_dim),
        #     # nn.ReLU(inplace=True),
        #     # nn.Dropout(p=0.2),
        # )
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.doubleconv = DoubleConv(in_offs, out_offs)

    def forward(self, x):
        # x = self.proj(x)
        # # proj: [B, C, patch_H, patch_W] -> [B, offset_num, patch_H, patch_W]
        #
        # x = x.flatten(2)
        # # flatten: [B, offset_num, patch_H, patch_W] -> [B, offset_num, patch_H*patch_W]
        #
        # x = self.fc_layer(x)
        # # fc_layer: [B, offset_num, patch_H*patch_W] -> [B, offset_num, embed_dim]
        #
        # _, patch_num, embed_dim = x.shape
        # x = x.permute(0, 2, 1)
        # # transpose: [B, offset_num, embed_dim] -> [B, embed_dim, offset_num]
        #
        # # x = self.norm(x)  # todo
        # x = x.reshape([-1, self.embed_dim, self.patches_resolution[0], self.patches_resolution[0]])
        # # reshape: [B, embed_dim, offset_num] -> [B, embed_dim, Wh, Ww]
        x = self.doubleconv(x)
        return x


class ConvMerging(nn.Module):
    r""" Conv Merging Layer.
    Args:
        dim (int): Number of input channels.
        out_dim (int): Output channels after the merging layer.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        # 使用卷积层进行下采样
        # self.reduction = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # 使用池化层进行下采样
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
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
        x_reshape = x

        x = self.norm(x)
        # B, C, H, W -> B, H*W, C
        x = self.reduction(x)
        # reduction: [B, C, Wh, Ww] -> [B, c(2C), Wh/2, Ww/2]
        B, C_out, H_out, W_out = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # flatten: [B, C_out(2C), Wh/2, Ww/2] -> [B, C_out(2C), Wh/2*Ww/2]
        # transpose: [B, C_out(2C), Wh/2*Ww/2] -> [B, Wh/2*Ww/2, C_out(2C)]
        return x, x_reshape, H_out, W_out


class BasicLayer(nn.Module):
    """ A basic layer for one stage in Mixing Blcok.
    Modified from Mixformer and Swin Transfomer BasicLayer.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end
            of the layer. Default: None
        out_dim (int): Output channels for the downsample layer. Default: 0.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 dwconv_kernel_size=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 out_dim=0,
                 val1=False,
                 val2=False):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            MixingBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                shift_size=0,
                # shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                val1=val1,
                val2=val2)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, None)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return H, W, x_down, Wh, Ww
        else:
            return H, W, x, H, W


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class MixUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,

                 patch_size=32,
                 ape=False,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=4,
                 dwconv_kernel_size=2,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 val1=False,
                 val2=False,
                 ):
        super(MixUNet, self).__init__()
        # UNet
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.base_channels = base_c

        # Mixing Block
        self.num_layers = len(depths)
        if isinstance(base_c, int):
            embed_dim = [base_c * 2 ** i_layer for i_layer in range(self.num_layers)]
            # embed_dim = [base_c, base_c*2, base_c*4, base_c*8]
        assert isinstance(embed_dim, list) and len(embed_dim) == self.num_layers
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm

        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim[-1])
        self.mlp_ratio = mlp_ratio

        # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        #     # self.add_parameter( "absolute_pos_embed", self.absolute_pos_embed)  #todo
        #     nn.init.trunc_normal_(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                val1=val1,
                val2=val2,
                dim=int(self.embed_dim[i_layer]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,  # 这里的下采样在U形结构的down中
                out_dim=int(self.embed_dim[i_layer + 1]) if (i_layer < self.num_layers - 1) else 0)
            self.layers.append(layer)

        self.in_conv = ConvEmbed(in_offs=in_channels, out_offs=self.base_channels, patch_size=patch_size)
        self.down1 = ConvMerging(base_c, base_c * 2)
        self.down2 = ConvMerging(base_c * 2, base_c * 4)
        self.down3 = ConvMerging(base_c * 4, base_c * 8)
        self.down4 = ConvMerging(base_c * 8, base_c * 16)
        factor = 2 if bilinear else 1
        self.mid_conv = DoubleConv(base_c * 16, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward_unet(self, x):

        _, _, Wh, Ww = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # x: [B, embed_dim, Wh, Ww] -> [B, Wh*Ww, embed_dim]

        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        layer_1 = self.layers[0]
        layer_2 = self.layers[1]
        layer_3 = self.layers[2]
        layer_4 = self.layers[3]
        H, W, x, Wh, Ww = self.layers[0](x, Wh, Ww)
        x, x1, Wh, Ww = self.down1(x, Wh, Ww)

        H, W, x, Wh, Ww = layer_2(x, Wh, Ww)
        x, x2, Wh, Ww = self.down2(x, Wh, Ww)

        H, W, x, Wh, Ww = layer_3(x, Wh, Ww)
        x, x3, Wh, Ww = self.down3(x, Wh, Ww)

        H, W, x, Wh, Ww = layer_4(x, Wh, Ww)
        x, x4, Wh, Ww = self.down4(x, Wh, Ww)

        B, L, C = x.shape
        assert L == Wh * Ww, "input feature has wrong size"
        # assert Wh % 2 == 0 and Ww % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.reshape([B, Wh, Ww, C]).permute(0, 3, 1, 2)

        x5 = self.mid_conv(x)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.in_conv(x)
        # in_conv：x:[B, 41, H, W] -> [B, base_c, H, W]
        x = self.forward_unet(x)
        # forward_unet：x:[B, base_c, H, W] -> [B, base_c, H, W]
        logits = self.out_conv(x)
        # forward_unet：x:[B, base_c, H, W] -> [B, num_classes, H, W]

        return {"out": logits}


if __name__ == '__main__':
    drop_path = 0.
    model = MixUNet(in_channels=41, num_classes=2 + 1,
                    bilinear=False,
                    base_c=64,
                    patch_size=32,
                    depths=(2, 2, 4, 2),
                    num_heads=(2, 4, 8, 16),
                    window_size=4,
                    dwconv_kernel_size=3,
                    mlp_ratio=2.,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                    val1=False,
                    val2=False)
    inputs = torch.randn((83, 41, 32, 32))
    outputs = model(inputs)
    print(outputs)
    output = outputs.get('out')
    print(output.shape)
    print(outputs['out'].shape)
