import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
        nn.ReLU(inplace=True)
    )

def stn(x, theta, padding_mode='zeros'):
    grid = F.affine_grid(theta, x.size())
    img = F.grid_sample(x, grid, padding_mode=padding_mode)

    return img


def _output_size_conv2d(conv, size):
    """
    Computes the output size of the convolution for an input size
    """
    o_size = np.array(size) + 2 * np.array(conv.padding)
    o_size -= np.array(conv.dilation) * (np.array(conv.kernel_size) - 1)
    o_size -= 1
    o_size = o_size / np.array(conv.stride) + 1
    return np.floor(o_size)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()

        self.in_ch = in_ch

        self.conv1 = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        out += residual
        return out

class BaseBlock(nn.Module):
    """
    Blocks for a residual model for reinforcement learning task as presented in He. and al, 2016
    """

    def __init__(self, in_ch, out_ch):
        super(BaseBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Conv2d(
            in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=3, stride=1, padding=1
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.residual = ResidualBlock(in_ch=self.out_ch)

        self._body = nn.Sequential(self.conv, self.max_pool, self.residual)

    def forward(self, x):
        x = self._body(x)
        return x

    def output_size(self, size):
        size = _output_size_conv2d(self.conv, size)
        size = _output_size_conv2d(self.max_pool, size)
        return size

class ImpalaConv(nn.Module):
    """
    Deeper model that uses 12 convolutions with residual blocks
    """

    def __init__(self, c):
        """c is the number of channels in the input tensor"""
        super(ImpalaConv, self).__init__()

        self.block_1 = BaseBlock(c, 16)
        self.block_2 = BaseBlock(16, 32)
        self.block_3 = BaseBlock(32, 64)

        self._body = nn.Sequential(self.block_1, self.block_2, self.block_3)

    def forward(self, x):
        return self._body(x)

    def output_size(self, size):
        size = self.block_1.output_size(size)
        size = self.block_2.output_size(size)
        size = self.block_3.output_size(size)
        return size


class Encoder_3d(nn.Module):
    def __init__(self, args):
        super(Encoder_3d, self).__init__()

        self.feature_extraction = ImpalaConv(3)  # 64 x 8 x 8

        self.depth_layer = 8
        self.feat_3d_ch = 64 // self.depth_layer  # z_2d: 512 x h/8 x w/8, z_3d: 512/(h/8) x h/8 x h/8 x h/8

        self.conv3d_1 = nn.ConvTranspose3d(self.feat_3d_ch, 8, 4, stride=2, padding=1) # 32 x 16 x 16 x 16
        self.conv3d_2 = nn.ConvTranspose3d(8, args.bottleneck, 3, stride=1, padding=1)  # 16 x 16 x 16 x 16


    def forward(self, img):
        # img -> 2d feature
        z_2d = self.feature_extraction(img)  # 128 x 8 x 8
        B, C, H, W = z_2d.shape  # 128 x 8 x 8
        # reshape
        z_3d = z_2d.reshape(
            [-1, self.feat_3d_ch, self.depth_layer, H, W])  # unproj  16x8x8x8

        # 3d convnet
        z_3d = self.conv3d_1(z_3d)  # 64 x 16 x 16 x 16
        z_3d = F.leaky_relu(z_3d)
        z_3d = self.conv3d_2(z_3d)  # bottleneck x 16 x 16 x 16
        z_3d = F.leaky_relu(z_3d)
        return z_3d


class Decoder_3d(nn.Module):
    def __init__(self, args):
        super(Decoder_3d, self).__init__()
        self.conv3 = nn.Conv2d(args.bottleneck * 16, args.bottleneck*8, 1)
        self.upconv3 = nn.ConvTranspose2d(args.bottleneck*8, args.bottleneck*8, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(args.bottleneck*8, 32, kernel_size=4, stride=2, padding=1)

        self.upconv_final = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, code):
        code = code.view(-1, code.size(1) * code.size(2), code.size(3), code.size(4))  # n x 16 x 16
        code = F.leaky_relu(self.conv3(code))  # 512 x 16 x 16

        # Decoder
        code = F.leaky_relu(self.upconv3(code))  # 256 x 32 x 32
        code = F.leaky_relu(self.upconv4(code))  # 32 x 64 x 64
        output = self.upconv_final(code)  # 3 x 128 x 128

        return output

class Rotate_3d(nn.Module):
    def __init__(self, args):
        super(Rotate_3d, self).__init__()
        self.padding_mode = "zeros"
        self.conv3d_1 = nn.Conv3d(args.bottleneck, args.bottleneck*2, 3, padding=1)
        self.conv3d_2 = nn.Conv3d(args.bottleneck*2, args.bottleneck, 3, padding=1)

    def forward(self, code, theta):  # 16 x 16 x 16 x 16
        rot_code = stn(code, theta, self.padding_mode)  # bottleneck x 16x16x16
        rot_code = F.leaky_relu(self.conv3d_1(rot_code))  # 32 x 16x16x16
        rot_code = F.leaky_relu(self.conv3d_2(rot_code))  # bottleneck x 16x16x16
        return rot_code

class Posenet_3d(nn.Module):
    def __init__(self):
        super(Posenet_3d, self).__init__()
        self.nb_ref_imgs = 1

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3 * (1 + self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])

        self.pose_pred = nn.Conv2d(conv_planes[2], 6 * self.nb_ref_imgs, kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, input):
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)

        pose = self.pose_pred(out_conv3)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), 6)

        return pose

