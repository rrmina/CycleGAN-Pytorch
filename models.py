import torch
import torch.nn as nn

class Generator(nn.Module):
    """Feedforward Transformation Network with Tanh
    reference: https://arxiv.org/abs/1603.08155 
    exact architecture: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
                        https://arxiv.org/pdf/1703.10593.pdf Appendix 7
    """
    def __init__(self, conv_dim):
        super(Generator, self).__init__()
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
            DeconvLayer(c*4, c*2, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(c*2, c, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(c, 3, 7, 1, norm="None"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        x = self.DeconvBlock(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, conv_dim, patch=True):
        super(Discriminator, self).__init__()
        c = conv_dim
        self.feature = nn.Sequential(
            ConvLayer(3, c, 4, 2, norm="None"),     # (3, 128) -> (c, 64) 
            nn.LeakyReLU(0.2),
            ConvLayer(c, c*2, 4, 2),                # (c, 64) -> (c*2, 32)
            nn.LeakyReLU(0.2),
            ConvLayer(c*2, c*4, 4, 2),              # (c*2, 32) -> (c*4, 16)
            nn.LeakyReLU(0.2),
            ConvLayer(c*4, c*8, 4, 2),              # (c*4, 8) -> (c*8, 8)
            nn.LeakyReLU(0.2),
        )
        if (patch): # Patch-GAN - Multiple discriminator output
            self.classifier = nn.Sequential( nn.Conv2d(c*8, 1, 4, 1) )
        else: # Original GAN Discrimnator - one scalar output
            self.classifier = nn.Sequential( nn.Conv2d(c*8, 1, 8, 1) ) # Assuming 128x128 input

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance", padding="reflection"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        self.padding = padding
        padding_size = kernel_size // 2
        if (padding == "reflection"):
            self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        if (padding == "None"):
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding_size, bias=False)
        else:
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)

        # Normalization Layer
        self.norm_type = norm
        if (norm == "instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm =="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        # Reflection Padding
        if (self.padding == "reflection"):
            x = self.reflection_pad(x)

        # Conv Layer
        x = self.conv_layer(x)

        # Norm Layer
        if (self.norm_type == "None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

class ResidualLayer(nn.Module):
    """
    Deep Residual Learning for Image Recognition

    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, channels=128, kernel_size=3):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)

    def forward(self, x):
        identity = x                     # preserve residual
        out = self.relu(self.conv1(x))   # 1st conv layer + activation
        out = self.conv2(out)            # 2nd conv layer
        out = out + identity             # add residual
        return out

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution
        padding_size = kernel_size //2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)
    
        # Normlaization Layer
        self.norm_type = norm
        if (norm == "instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm =="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out