import torch
import torch.nn as nn
from models import ConvLayer, ResidualLayer

class GeneratorUpsampleConv(nn.Module):
    """Feedforward Transformation Network with Tanh
    reference: https://arxiv.org/abs/1603.08155 
    exact architecture: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
                        https://arxiv.org/pdf/1703.10593.pdf Appendix 7
    """
    def __init__(self, conv_dim):
        super(GeneratorUpsampleConv, self).__init__()
        c = conv_dim
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, c, 7, 1),
            nn.ReLU(),
            ConvLayer(c, c*2, 3, 2),
            nn.ReLU(),
            ConvLayer(c*2, c*4, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(c*4, 3),
            ResidualLayer(c*4, 3),
            ResidualLayer(c*4, 3),
            ResidualLayer(c*4, 3),
            ResidualLayer(c*4, 3),
            ResidualLayer(c*4, 3),
        )
        self.DeconvBlock = nn.Sequential(
            UpsampleConvLayer(c*4, c*2, kernel_size=3, stride=1, upsample=2),
            nn.ReLU(),
            UpsampleConvLayer(c*2, c, kernel_size=3, stride=1, upsample=2),
            nn.ReLU(),
            ConvLayer(c, 3, 7, 1, norm="None"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        x = self.DeconvBlock(x)
        return x

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    code_ref: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
