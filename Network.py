import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.distributions as td
from torch.amp import autocast

# Third-party libraries
from mamba_ssm import Mamba
from einops import rearrange, repeat
import timm
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from timm.models.layers import DropPath, trunc_normal_

# Python standard libraries
import time
import math
import copy
from functools import partial
from typing import Optional, Callable
    
import torch
import torch.nn as nn
import torch
import torch.nn as nn

from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        drop = 0.
        self.fc1 = nn.Conv3d(dim, dim * 4, 1)
        self.dwconv = nn.Conv3d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(dim * 4, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class MLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.att_conv2 = nn.Conv3d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv3 = nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.skf = MSDABlock(dim,height=3)


    def forward(self, x):   
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)+att1
        att3 = self.att_conv3(att2)+att2+att1
        att_res = self.skf([att1,att2,att3])
        return att_res + x




class AdaptiveMeanAndStdPool3d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size)
        
    def forward(self, x):
        mean = self.avg_pool(x)
        squared = self.avg_pool(x * x)
        # sqrt(E(X^2) - E(X)^2)
        std = torch.sqrt(torch.clamp(squared - mean * mean, min=1e-10))
        return mean,std
    
    
class MSDABlock(nn.Module):
    def __init__(self, channel,height=2,kernel_sizes=[3, 5, 7]):
        super(MSDABlock, self).__init__()
        
        self.height = height
        self.kernel_sizes = kernel_sizes
        self.avg_pool = AdaptiveMeanAndStdPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fusion = nn.Linear(height*3*channel,channel*height)
        self.fusion_conv = nn.Sequential(nn.Conv3d(channel*3,channel,1,1,0),
                nn.GroupNorm(num_groups=channel,num_channels=channel),
                nn.LeakyReLU(inplace=True)
                )
        self.fusion_res = nn.Sequential(nn.Conv3d(channel*3,channel,1,1,0),
                nn.GroupNorm(num_groups=channel,num_channels=channel),
                nn.LeakyReLU(inplace=True)
                )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, in_feats):
        B, C, D, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height*C, D, H, W) 
        avg_attn,std_attn = self.avg_pool(in_feats) 
        max_attn = self.max_pool(in_feats) 
        combined_attn = torch.cat([avg_attn, std_attn, max_attn], dim=1)
        combined_attn = combined_attn.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        attn = self.fusion(combined_attn)
        attn = attn.squeeze(1)
        attn = self.softmax(attn).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = self.fusion_conv(in_feats * attn)
        return out + self.fusion_res(in_feats)

class MLKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv3d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = MLK(dim)
        self.proj_2 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

def channel_to_last(x):
    """
    Args:
        x: (B, C, H, W, D)

    Returns:
        x: (B, H, W, D, C)
    """
    return x.permute(0, 2, 3, 4, 1)


def channel_to_first(x):
    """
    Args:
        x: (B, H, W, D, C)

    Returns:
        x: (B, C, H, W, D)
    """
    return x.permute(0, 4, 1, 2, 3)
class MLKBlock(nn.Module):
    def __init__(self, dim,output_dim, drop_path=0.):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MLKModule(dim)
        self.mlp = Mlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-6         
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.up_channel_conv = nn.Conv3d(
            dim,
            output_dim,
            kernel_size=1,
            stride=1)
    def forward(self, x):
        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x)

        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x)
        return self.up_channel_conv(x)



class DenseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=4,
        dropout_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.conv_list = nn.ModuleList()

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
                nn.InstanceNorm3d(in_channels, affine=True),
                # LayerNormBatchFirst(in_channels),
                # nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels + in_channels, in_channels * expand_rate, 1, 1, 0
                ),
                # nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                # nn.LeakyReLU(inplace=True),
                nn.GELU(),
            )
        )
        temp_in_channels = in_channels + in_channels + in_channels * expand_rate
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, out_channels, 1, 1, 0),
                # nn.InstanceNorm3d(out_channels, affine=True),
                # nn.LeakyReLU(inplace=True),
            )
        )
        self.dp_1 = nn.Dropout(dropout_rate)
        self.dp_2 = nn.Dropout(dropout_rate * 2)
        self.residual = in_channels == out_channels
        self.drop_path = (
            True if torch.rand(1) < drop_path_rate and self.training else False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        if self.drop_path and self.residual:
            return res
        x1 = self.conv_list[0](x)
        x1 = self.dp_1(x1)
        x2 = self.conv_list[1](torch.cat([x, x1], dim=1))
        x2 = self.dp_2(x2)
        x = (
            self.conv_list[2](torch.cat([x, x1, x2], dim=1)) + res
            if self.residual
            else self.conv_list[2](torch.cat([x, x1, x2], dim=1))
        )
        return x


class DenseConvDown(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=4,
        dropout_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.conv_list = nn.ModuleList()
        self.res = in_channels == out_channels
        self.dense = stride == 1

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
                nn.GroupNorm(num_groups=in_channels//8,num_channels=in_channels),
                nn.LeakyReLU(inplace=True),
            )
        )
        temp_in_channels = in_channels + in_channels if self.dense else in_channels
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, in_channels * expand_rate, 1, 1, 0),
                nn.GroupNorm(num_groups=in_channels * expand_rate//8,num_channels=in_channels * expand_rate),
                nn.LeakyReLU(inplace=True),
            )
        )
        temp_in_channels = (
            in_channels + in_channels + in_channels * expand_rate
            if self.dense
            else in_channels + in_channels * expand_rate
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, out_channels, 1, 1, 0),
                nn.GroupNorm(num_groups=out_channels//8,num_channels=out_channels),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.dp_1 = nn.Dropout(dropout_rate)
        self.dp_2 = nn.Dropout(dropout_rate * 2)
        self.drop_path = (
            True if torch.rand(1) < drop_path_rate and self.training else False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        if self.drop_path and self.dense:
            return res
        x1 = self.conv_list[0](x)
        x1 = self.dp_1(x1)
        x2 = (
            self.conv_list[1](torch.cat([x, x1], dim=1))
            if self.dense
            else self.conv_list[1](x1)
        )
        x2 = self.dp_2(x2)
        if self.dense and self.res:
            x = self.conv_list[2](torch.cat([x, x1, x2], dim=1)) + res
        elif self.dense and not self.res:
            x = self.conv_list[2](torch.cat([x, x1, x2], dim=1))
        else:
            x = self.conv_list[2](torch.cat([x1, x2], dim=1))
        return x


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_conv=1,
        conv=DenseConvDown,
        stride=2,
        **kwargs,
    ):
        super().__init__()
        self.downsample_avg = nn.AvgPool3d(stride)
        self.downsample_densenet = DenseConvDown(in_channels, in_channels,stride=stride)
        # self.projection_conv = nn.Conv3d(in_channels, in_channels,kernel_size=1,stride=1,padding=0)
        # self.projection_mamba = nn.Conv3d(in_channels, in_channels,kernel_size=1,stride=1,padding=0)
        # self.mschead = AxialAttention3D(in_channels=in_channels)
        self.up = nn.Conv3d(in_channels,
            out_channels,
            kernel_size=1,
            stride=1)
        self.DLKBlock = MLKBlock(in_channels, out_channels)
        self.extractor = nn.ModuleList(
            [
                (
                    conv(in_channels, out_channels, **kwargs)
                    if _ == 0
                    else conv(out_channels, out_channels, **kwargs)
                )
                for _ in range(num_conv)
            ]
        )

    def forward(self, x):
        # x = self.downsample_avg(x)+self.downsample_resnext(x)+self.res_layer(x)
        x = self.downsample_avg(x) + self.downsample_densenet(x)
        # x1 = self.projection_conv(x)
        # x2 = self.projection_mamba(x)
        # x_temp = torch.cat([self.Fmamba(x1), self.mschead(x2)],dim=1)
        x_fusion = self.DLKBlock(x)
        # return x_fusion
        return self.up(x) + x_fusion


class Up(nn.Module):
    def __init__(
        self,
        low_channels,
        high_channels,
        out_channels,
        num_conv=1,
        conv=DenseConv,
        fusion_mode="add",
        stride=2,
        **kwargs,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.up_transpose = nn.ConvTranspose3d(
            low_channels, high_channels, stride, stride
        )
        in_channels = 2 * high_channels if fusion_mode == "cat" else high_channels
        self.extractor = nn.ModuleList(
            [
                (
                    conv(in_channels, out_channels)
                    if _ == 0
                    else conv(out_channels, out_channels)
                )
                for _ in range(num_conv)
            ]
        )

    def forward(self, x_low, x_high):
        x_low = self.up_transpose(x_low)
        x = (
            torch.cat([x_high, x_low], dim=1)
            if self.fusion_mode == "cat"
            else x_low + x_high 
        )
        for extractor in self.extractor:
            x = extractor(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.1):
        super().__init__()
        self.dp = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv3d(in_channels, num_classes, 1, 1, 0)

    def forward(self, x):
        x = self.dp(x)
        p = self.conv1(x)
        return p




class BFF(nn.Module):
    def __init__(self, sizes_list, channels_list, use_deconv=True):
        super(BFF, self).__init__()
        
        self.sizes_list = sizes_list
        self.channels_list = channels_list
        self.use_deconv = use_deconv
        
        if use_deconv:
            self.upsample_layers = nn.ModuleList([
                nn.ConvTranspose3d(
                    channels_list[i+1], 
                    channels_list[i+1], 
                    kernel_size=2, 
                    stride=2
                )
                for i in range(len(channels_list)-1)
            ])
        
        self.channel_adjusts = nn.ModuleList([
            nn.Conv3d(channels_list[i+1], channels_list[i], kernel_size=1)
            for i in range(len(channels_list)-1)
        ])

    def forward(self, features):

        fused_features = [features[-1]]  # 
        
        # 
        for i in range(len(features)-2, -1, -1):
            deeper_feat = fused_features[-1]
            current_feat = features[i]
            
            if self.use_deconv:
                deeper_feat = self.upsample_layers[i](deeper_feat)
            else:
                deeper_feat = nn.functional.interpolate(
                    deeper_feat,
                    size=self.sizes_list[i],
                    mode='trilinear',
                    align_corners=True
                )
            
            deeper_feat = self.channel_adjusts[i](deeper_feat)
            fused = current_feat + deeper_feat
            
            fused_features.append(fused)
        
        return fused_features[::-1]  # 


class DBFUNET(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth=4,
        conv=DenseConv,
        channels=[2**i for i in range(4, 9)],
        encoder_num_conv=[0, 0, 0, 0],
        decoder_num_conv=[1, 1, 1, 1],
        encoder_expand_rate=[4] * 4,
        decoder_expand_rate=[4] * 4,
        strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        dropout_rate_list=[0.025, 0.05, 0.1, 0.1],
        drop_path_rate_list=[0.025, 0.05, 0.1, 0.1],
        deep_supervision=False,
        predict_mode=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.predict_mode = predict_mode

        assert len(channels) == depth + 1, "len(encoder_channels) != depth + 1"
        assert len(strides) == depth, "len(strides) != depth"

        self.encoders = nn.ModuleList()  #
        self.decoders = nn.ModuleList()  #
        self.encoders.append(DenseConv(in_channels, channels[0]))        
        self.DLKTopSKC = MLKBlock(channels[0], channels[0])
        # self.encoders.append(SpatialRotation3D(in_channels, channels[0]))
        self.skip = nn.ModuleList()  # 
        for i in range(self.depth):
            self.encoders.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    conv=conv,
                    num_conv=encoder_num_conv[i],
                    stride=strides[i],
                    expand_rate=encoder_expand_rate[i],
                    dropout_rate=dropout_rate_list[i],
                    drop_path_rate=drop_path_rate_list[i],
                ),
            )
        for i in range(self.depth):
            self.decoders.append(
                Up(
                    low_channels=channels[self.depth - i],
                    high_channels=channels[self.depth - i - 1],
                    out_channels=channels[self.depth - i - 1],
                    # conv=conv,
                    num_conv=decoder_num_conv[self.depth - i - 1],
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    expand_rate=decoder_expand_rate[self.depth - i - 1],
                    dropout_rate=dropout_rate_list[self.depth - i - 1],
                    drop_path_rate=drop_path_rate_list[self.depth - i - 1],
                )
            )
            

        self.out = nn.ModuleList(
            [Out(channels[depth - i - 1], n_classes) for i in range(depth)]
        )
        sizes_list = [(128,128,128),(64,64,64), (32,32,32), (16,16,16), (16,16,16)]

        # channels_list = [64, 128, 256, 512]
        channels_list = [16,32, 64, 128, 256]
        self.skc = BFF(sizes_list,channels_list)
        
    def forward(self, x):
        encoder_features = []
        decoder_features = [] 

        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        x_high = encoder_features[0]
        encoder_features=self.skc(encoder_features[:])
        # encoder_features.insert(0,x_high)
        encoder_features[0] = encoder_features[0] + x_high
        for i in range(self.depth+1):
            if i == 0:
                x_dec =  encoder_features[-1]
            elif i == self.depth:
                x_dec = self.decoders[i-1](x_dec, encoder_features[-(i + 1)]) + self.DLKTopSKC(encoder_features[-(i + 1)])
                decoder_features.append(x_dec)
            else:
                x_dec = self.decoders[i-1](x_dec, encoder_features[-(i + 1)])
                decoder_features.append(x_dec)
        
        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        elif self.predict_mode:
            return self.out[-1](decoder_features[-1])
        else:
            return x_dec, self.out[-1](decoder_features[-1])



if __name__ == "__main__":
    model = DBFUNET(1, 2).to("cuda:6")
    # x = torch.randn(2, 1, 128, 128, 128).to("cuda:6")
    # y = model(x)
    # print(y.shape)
    # import time
    # time.sleep(10000)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (1,128, 128, 128), as_strings=True, print_per_layer_stat=True, verbose=True)

    print(f'Computational complexity: {macs}')
    print(f'Number of parameters: {params}')
    # print(summary(model, input_size=(2, 1, 128, 128, 128), device="cuda:6", depth=5))
