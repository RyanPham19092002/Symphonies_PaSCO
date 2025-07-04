# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import time
from pasco.models.metrics import SSCMetrics

# Must be imported before large libs
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

import MinkowskiEngine as ME
from pasco.maskpls.mink import BasicConvolutionBlock, ResidualBlock


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer,
        act_layer,
        dropout_layer,
        drop_path_rates=[0.0, 0.0, 0.0],
        downsample=True,
        use_se_layer=False,
        dropout=0.0,
    ) -> None:
        super().__init__()
        print("enc use_se_layer", use_se_layer)
        self.downsample = downsample
        if self.downsample:
            # self.down = nn.Sequential(
            #    DepthwiseSeparableConvMultiheadsV2(
            #         in_channels, out_channels,
            #         n_heads=n_heads,
            #         kernel_size=2, stride=2
            #     ),
            #     norm_layer(out_channels),
            #     act_layer(),
            # )
            self.down = nn.Sequential(
                BasicConvolutionBlock(in_channels, out_channels, ks=2, stride=2),
                norm_layer(out_channels),
                act_layer(),
            )
            self.conv = nn.Sequential(
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[0]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[1]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[2]),
                dropout_layer(p=dropout),
            )
        else:
            self.conv = nn.Sequential(
                ResidualBlock(out_channels, out_channels),
                ResidualBlock(out_channels, out_channels),
                ResidualBlock(out_channels, out_channels),
                dropout_layer(p=dropout),
            )

    def forward(self, x):
        if self.downsample:
            x = self.down(x)
        return self.conv(x)


class Encoder3DSepV2(nn.Module):

    def __init__(
        self,
        in_channels,
        f,
        norm_layer,
        act_layer,
        dropout_layer,
        dropouts,
        heavy_decoder=True,
        drop_path_rates=None,
        n_heads=1,
        use_se_layer=False,
    ):
        nn.Module.__init__(self)
        # Input sparse tensor must have tensor stride 128.
        enc_ch = f
        # enc_ch = [x * 2 for x in f]
        self.enc_in_feats = ME.MinkowskiConvolution(
            in_channels, enc_ch[0], kernel_size=1, stride=1, dimension=3
        )

        self.enc_img_feats = ME.MinkowskiConvolution(
            in_channels, enc_ch[0], kernel_size=1, stride=1, dimension=3
        )

        if drop_path_rates is None:
            drop_path_rates = [0.0] * 12

        if not heavy_decoder:
            
            self.s1_img = nn.Sequential(
                ResidualBlock(enc_ch[0], enc_ch[0]),
                ResidualBlock(enc_ch[0], enc_ch[0]),
                ResidualBlock(enc_ch[0], enc_ch[0]),
                nn.Identity(),
            )

            self.s1s2_img = nn.Sequential(
                BasicConvolutionBlock(enc_ch[0], enc_ch[1], ks=2, stride=2),
                norm_layer(enc_ch[1]),
                act_layer(),
                ResidualBlock(enc_ch[1], enc_ch[1]),
                ResidualBlock(enc_ch[1], enc_ch[1]),
                ResidualBlock(enc_ch[1], enc_ch[1]),
            )

            self.s2s4_img = nn.Sequential(
                BasicConvolutionBlock(enc_ch[1], enc_ch[2], ks=2, stride=2),
                norm_layer(enc_ch[2]),
                act_layer(),
                ResidualBlock(enc_ch[2], enc_ch[2]),
                ResidualBlock(enc_ch[2], enc_ch[2]),
                ResidualBlock(enc_ch[2], enc_ch[2]),
            )
            
            self.s4s8_img = nn.Sequential(
                BasicConvolutionBlock(enc_ch[2], enc_ch[3], ks=2, stride=2),
                norm_layer(enc_ch[3]),
                act_layer(),
                ResidualBlock(enc_ch[3], enc_ch[3]),
                ResidualBlock(enc_ch[3], enc_ch[3]),
                ResidualBlock(enc_ch[3], enc_ch[3]),
            )
            self.s1 = nn.Sequential(
                ResidualBlock(enc_ch[0], enc_ch[0]),
                ResidualBlock(enc_ch[0], enc_ch[0]),
                ResidualBlock(enc_ch[0], enc_ch[0]),
                nn.Identity(),
            )
            self.s1s2 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[0]*2, enc_ch[1], ks=2, stride=2),
                norm_layer(enc_ch[1]),
                act_layer(),
                ResidualBlock(enc_ch[1], enc_ch[1]),
                ResidualBlock(enc_ch[1], enc_ch[1]),
                ResidualBlock(enc_ch[1], enc_ch[1]),
            )

            self.s2s4 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[1]*2, enc_ch[2], ks=2, stride=2),
                norm_layer(enc_ch[2]),
                act_layer(),
                ResidualBlock(enc_ch[2], enc_ch[2]),
                ResidualBlock(enc_ch[2], enc_ch[2]),
                ResidualBlock(enc_ch[2], enc_ch[2]),
            )

            self.s4s8 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[2]*2, enc_ch[3], ks=2, stride=2),
                norm_layer(enc_ch[3]),
                act_layer(),
                ResidualBlock(enc_ch[3], enc_ch[3]),
                ResidualBlock(enc_ch[3], enc_ch[3]),
                ResidualBlock(enc_ch[3], enc_ch[3]),
            )

            self.final_s8 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[3]*2, enc_ch[3], ks=1, stride=1, dilation=1),
                norm_layer(enc_ch[3]),
                act_layer()
            )
        else:
            self.s1 = nn.Sequential(
                nn.Identity(),
            )
            self.s1s2 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[0], enc_ch[1], ks=2, stride=2),
                norm_layer(enc_ch[1]),
                act_layer(),
                dropout_layer(p=dropouts[-3]),
            )

            self.s2s4 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[1], enc_ch[2], ks=2, stride=2),
                norm_layer(enc_ch[2]),
                act_layer(),
                dropout_layer(p=dropouts[-2]),
            )

            self.s4s8 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[2], enc_ch[3], ks=2, stride=2),
                norm_layer(enc_ch[3]),
                act_layer(),
                dropout_layer(p=dropouts[-1]),
            )

    def forward(self, in_feats, img_feats):
        # start_time = time.perf_counter()
        partial_in = self.enc_in_feats(in_feats)
        # partial_in_time = time.perf_counter()
        # print(f"Encoder3DSepV2: enc_in_feats time: {partial_in_time - start_time:.4f} seconds")
        partial_img = self.enc_img_feats(img_feats)
        # partial_img_time = time.perf_counter()
        # print(f"Encoder3DSepV2: enc_img_feats time: {partial_img_time - partial_in_time:.4f} seconds")
        # enc_partial_concate = ME.cat((partial_in, partial_img))
        
        enc_s1_fused = self.s1(partial_in)
        # enc_s1_fused_time = time.perf_counter()
        # print(f"Encoder3DSepV2: s1 time: {enc_s1_fused_time - partial_img_time:.4f} seconds")
        enc_img_s1 = self.s1_img(partial_img)    
        # enc_img_s1_time = time.perf_counter()
        # print(f"Encoder3DSepV2: s1_img time: {enc_img_s1_time - enc_s1_fused_time:.4f} seconds")
        enc_s1_concate = ME.cat((enc_s1_fused, enc_img_s1))

        enc_s2_fused = self.s1s2(enc_s1_concate)
        # enc_s2_fused_time = time.perf_counter()
        # print(f"Encoder3DSepV2: s1s2 time: {enc_s2_fused_time - enc_img_s1_time:.4f} seconds")
        enc_img_s2 = self.s1s2_img(enc_img_s1)
        # enc_img_s2_time = time.perf_counter()
        # print(f"Encoder3DSepV2: s1s2_img time: {enc_img_s2_time - enc_s2_fused_time:.4f} seconds")
        enc_s2_concate = ME.cat((enc_s2_fused, enc_img_s2))
        # enc_s4 = self.s2s4(enc_s2)
        enc_s4_fused = self.s2s4(enc_s2_concate)
        # enc_s4_fused_time = time.perf_counter()
        # print(f"Encoder3DSepV2: s2s4 time: {enc_s4_fused_time - enc_img_s2_time:.4f} seconds")
        enc_img_s4 = self.s2s4_img(enc_img_s2)
        # enc_img_s4_time = time.perf_counter()
        # print(f"Encoder3DSepV2: s2s4_img time: {enc_img_s4_time - enc_s4_fused_time:.4f} seconds")
        enc_s4_concate = ME.cat((enc_s4_fused, enc_img_s4))
        # enc_s8 = self.s4s8(enc_s4)
        enc_s8_fused = self.s4s8(enc_s4_concate)
        # enc_s8_fused_time = time.perf_counter()
        # print(f"Encoder3DSepV2: s4s8 time: {enc_s8_fused_time - enc_img_s4_time:.4f} seconds")
         # enc_img_s8 = self.s4s8_img(enc_img_s4)
        enc_img_s8 = self.s4s8_img(enc_img_s4)
        # enc_img_s8_time = time.perf_counter()
        # print(f"Encoder3DSepV2: s4s8_img time: {enc_img_s8_time - enc_s8_fused_time:.4f} seconds")
        enc_s8_concate = ME.cat((enc_s8_fused, enc_img_s8))

        enc_s8_final = self.final_s8(enc_s8_concate)
        # enc_s8_final_time = time.perf_counter()
        # print(f"Encoder3DSepV2: final_s8 time: {enc_s8_final_time - enc_img_s8_time:.4f} seconds")
        # print(f"Encoder3DSepV2: total time: {enc_s8_final_time - start_time:.4f} seconds")

        features = [enc_s1_fused, enc_s2_fused, enc_s4_fused, enc_s8_final]

        return features, enc_img_s8
