import torch.nn as nn


# Architecture for network that learns "style", generating stylized version of input image.
# Architecture pretty much copied from {Johnson/Fei Fei Li paper}, with some improvements (instance norm and
# reflection pad) implemented as suggested by {Google paper?}

# Code itself was adapted from implementation in: https://github.com/rrmina/fast-neural-style-pytorch
class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU(),
            ConvLayer(128, 128, 1, 1),  # ADDED CONV LAYER WITH 1 x 1 KERNEL
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 128, 1, 1, 0),  # ADDED CONV LAYER WITH 1 x 1 KERNEL
            nn.ReLU(),
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        if kernel_size > 1:
            # Padding Layers
            padding_size = kernel_size // 2
            self.reflection_pad = nn.ReflectionPad2d(padding_size)
        else:
            self.reflection_pad = nn.Identity()

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if self.norm_type == "None":
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
        identity = x  # preserve residual
        out = self.relu(self.conv1(x))  # 1st conv layer + activation
        out = self.conv2(out)  # 2nd conv layer
        out = out + identity  # add residual
        return out


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size,
                                                 output_padding)

        # Normalization Layers
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out

# THIS STUFF IS ORGANIZED MORE NICELY, BUT DOESN'T WORK WITH ALREADY-TRAINED MODELS
# class StyleTransfer(nn.Module):
#     def __init__(self):
#         super(StyleTransfer, self).__init__()
#         self.ConvBlock = nn.Sequential(
#             ConvLayer(3, 32, 9, 1),
#             ConvLayer(32, 64, 3, 2),
#             ConvLayer(64, 128, 3, 2),
#             ConvLayer(128, 128, 1, 1),  # ADDED CONV LAYER WITH 1 x 1 KERNEL
#         )
#         self.ResidualBlock = nn.Sequential(
#             ResidualLayer(128, 3),
#             ResidualLayer(128, 3),
#             ResidualLayer(128, 3),
#             ResidualLayer(128, 3),
#             ResidualLayer(128, 3)
#         )
#         self.DeConvBlock = nn.Sequential(
#             DeConvLayer(128, 128, 1, 1, 0),  # ADDED CONV LAYER WITH 1 x 1 KERNEL
#             DeConvLayer(128, 64, 3, 2, 1),
#             DeConvLayer(64, 32, 3, 2, 1),
#             ConvLayer(32, 3, 9, 1, use_relu=False, use_norm=False)
#         )
#
#     def forward(self, x):
#         x = self.ConvBlock(x)
#         x = self.ResidualBlock(x)
#         out = self.DeConvBlock(x)
#         return out
#
#
# # Implement reflection padding (before) and instance normalization (after)
# # in addition to the normal convolutional layer
# class ConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=True, use_norm=True):
#         super(ConvLayer, self).__init__()
#
#         # Convolution Layer
#         self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#
#         # ReLU
#         self.uses_relu = use_relu
#         self.relu = nn.ReLU()
#
#         # Instance normalization layer
#         self.uses_norm = use_norm
#         self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
#
#         # Padding layer
#         self.uses_pad = kernel_size > 1
#         self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
#
#     def forward(self, x):
#         if self.uses_pad:
#             x = self.reflection_pad(x)
#         x = self.conv_layer(x)
#         if self.uses_norm:
#             x = self.norm_layer(x)
#         if self.uses_relu:
#             x = self.relu(x)
#         return x
#
#
# # Implement residual layer, which is two convolutional layers of the same size whose output is added to the initial
# # input. This way, the layer maintains some of the information about the input in addition to the learned responses
# class ResidualLayer(nn.Module):
#     def __init__(self, channels=128, kernel_size=3):
#         super(ResidualLayer, self).__init__()
#         self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, use_relu=True)
#         self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, use_relu=False)
#
#     def forward(self, x):
#         identity = x  # preserve residual
#         out = self.conv1(x)  # 1st conv layer with ReLU
#         out = self.conv2(out)  # 2nd conv layer without ReLU
#         out = out + identity  # add residual
#         return out
#
#
# class DeConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding):
#         super(DeConvLayer, self).__init__()
#
#         # Transposed Convolution
#         self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
#                                                  padding=kernel_size // 2, output_padding=output_padding)
#
#         # Normalization Layer
#         self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
#
#         # ReLU activation
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv_transpose(x)
#         x = self.norm_layer(x)
#         out = self.relu(x)
#         return out
