from abc import abstractmethod

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


'''
DIFFUSION MODEL BASED ON THE FOLLOWING PAPERS AND CORRESPONDING WORK:
    - Ho et al. Denoising Diffusion Probabilistic Models (DDPM): https://arxiv.org/pdf/2006.11239.pdf
    - Song et al. Denoising Diffusion Implicit Models (DDIM): https://arxiv.org/pdf/2010.02502.pdf
    - Dhariwal/Nichol Diffusion Model Beats GAN on Image Synthesis (OpenAI): https://arxiv.org/pdf/2105.05233.pdf
    
    Model architecture closely follows DDPM's and OpenAI's, implementation largely same with some modifications for 
    clarity, convenience, and a different attention implementation. 
    
    I will also add classifier-dependent and classifier-free guidance as per Ho/Salismans
    https://openreview.net/pdf/ea628d03c92a49b54bc2d757d209e024e7885980.pdf once I get around to it :)
'''


# Abstract wrapper class to be used when forward propagation needs step embeddings
class UsesSteps(nn.Module):
    @abstractmethod
    def forward(self, x, step):
        """
        To be used when forward propagation needs step embeddings
        """


def override(a):  # silly function i had to write to use @override, sometimes python can be annoying lol
    return a


# Wrapper sequential class that knows when to pass time step embedding to its children or not
class UsesStepsSequential(nn.Sequential, UsesSteps):
    @override
    def forward(self, x, step):
        for layer in self:
            if isinstance(layer, UsesSteps):
                x = layer(x, step)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Creates a module whose forward pass performs 2x upsampling of input image.

        Parameters:
            - in_channels (int): number of channels to pass in
            - out_channels (int): number of channels to return. if None, return with in_channels channels
            - with_conv (bool)L if True use Conv2D for upsampling, if False use nearest neighbor interpolation


        Returns:
            - An nn.Module to be used to compose a network.
    """
    def __init__(self, in_channels, with_conv, out_channels=None):
        super(Upsample, self).__init__()
        self.with_conv = with_conv

        if with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels if out_channels is not None else in_channels,
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Creates a module whose forward pass performs 2x downsampling of input image.

        Parameters:
            - in_channels (int): number of channels to pass in
            - out_channels (int): number of channels to return. if None, return with in_channels channels
            - with_conv (bool): if True use Conv2D for downsampling, if False use 2D avg. pooling

        Returns:
            - An nn.Module to be used to compose a network.
    """
    def __init__(self, in_channels, with_conv, out_channels=None):
        super(Downsample, self).__init__()
        self.with_conv = with_conv

        if with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels if out_channels is not None else in_channels,
                                  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # ADD ASYMMETRIC PADDING??

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        return x


# Can be used with up or downsampling
# If use_conv=True and out_channels!=None, use spatial convolution to change number of channels
class ResidualBlock(UsesSteps):
    """
    Creates a residual block containing 2 convolutional layers with a skip connection.
    Uses timestep embeddings and 2 layers of GroupNorm (32 groups).

        Parameters:
            - in_channels (int): number of channels to pass in
            - out_channels (int): number of channels to return. if None, return with in_channels channels
            - step_channels (int): number of channels to use for timestep embedding
            - upsample (bool): whether to upsample before first convolution
            - downsample (bool): whether to downsample before first convolution
            - use_conv (bool): if out_channels != in_channels and this is True, use 3x3 convolution to
            - change number of channels
            - use_scale_shift_norm (bool): whether to use step embedding to scale and shift input or simply add them
            - dropout (double): dropout probability

        Returns:
            - An nn.Module to be used to compose a network.
    """
    def __init__(self, in_channels, step_channels, dropout, upsample=False, downsample=False,
                 use_conv=False, out_channels=None, use_scale_shift_norm=False):
        super(ResidualBlock, self).__init__()

        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.silu = nn.SiLU()
        out_channels = out_channels if out_channels is not None else in_channels

        if upsample:
            self.h_resample = Upsample(in_channels=in_channels, with_conv=False)
            self.x_resample = Upsample(in_channels=in_channels, with_conv=False)
            self.resample = True
        elif downsample:
            self.h_resample = Downsample(in_channels=in_channels, with_conv=False)
            self.x_resample = Downsample(in_channels=in_channels, with_conv=False)
            self.resample = True
        else:
            self.h_resample = self.x_resample = nn.Identity()
            self.resample = False

        # Change number of channels in skip connection if necessary
        if out_channels == in_channels:
            self.skip = nn.Identity()
        elif use_conv:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(1, 1), stride=(1, 1))

        self.in_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-5)
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-5)
        self.out_conv = zero_module(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        if use_scale_shift_norm:
            self.step_embedding = nn.Linear(step_channels, 2 * out_channels)
        else:
            self.step_embedding = nn.Linear(step_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, step):
        h = x
        h = self.silu(self.in_norm(h))
        if self.resample:
            h = self.h_resample(h)
            x = self.x_resample(x)
        h = self.in_conv(h)

        # add in timestep embedding
        embedding = self.step_embedding(self.silu(step))[:, :, None, None]
        # apply scale-shift norm
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(embedding, 2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
        else:
            h += embedding
            h = self.out_norm(h)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.out_conv(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Creates an attention-style Transformer block containing QKV-attention, a residual connection,
    and a linear projection out.
    Uses 1 layer of GroupNorm (32 groups).

        Parameters:
            - channels (int): number of channels to pass in and return out
            - num_heads (int): number of heads to use in Multi-Headed-Attention
            - num_head_channels (int): number of channels to use for each head. if not None, then this block uses
            (channels // num_head_channels) heads and ignores num_heads

        Returns:
            - An nn.Module to be used to compose a network.
    """
    def __init__(self, channels, num_heads=1, num_head_channels=None):
        super(AttentionBlock, self).__init__()

        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), "channels {} is not divisible by num_head_channels {}".format(channels, num_head_channels)
            self.num_heads = channels // num_head_channels

        self.scale = (channels // self.num_heads) ** -0.5

        self.qkv_nin = nn.Conv1d(in_channels=channels, out_channels=3 * channels,
                                 kernel_size=(1,), stride=(1,))

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-5)
        # self.proj_out = zero_module(nn.Conv1d(in_channels=channels, out_channels=channels,
        #                                       kernel_size=(1,), stride=(1,)))
        self.proj_out = zero_module(nn.Conv1d(in_channels=channels, out_channels=channels,
                                              kernel_size=(1,), stride=(1,)))

    # My implementation of MHA
    def forward(self, x):
        # compute 2-D attention (condense H,W dims into N)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        qkv = self.norm(x)

        # Get q, k, v
        qkv = self.qkv_nin(qkv).permute(0, 2, 1)  # b,c,hw -> b,hw,c
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # q/k/v.shape = b,num_heads,hw,c//num_heads

        # w = softmax(q @ k / sqrt(d_k))
        w = (q @ k.transpose(-2, -1)) * self.scale
        w = torch.softmax(w, dim=-1)

        # attention = w @ v
        h = (w @ v).transpose(1, 2).reshape(B, H * W, C).permute(0, 2, 1)
        h = self.proj_out(h)

        return (h + x).reshape(B, C, H, W)


class DiffusionModel(nn.Module):
    """
    Creates a Diffusion model to predict epsilon in a generative denoising process.

        Parameters:
            - resolution (int): height and width resolution of inputs (assumes square images)
            - in_channels (int): number of channels to pass in
            - out_channels (int): number of channels to return
            - model_channels (int): number of channels to use within the model, before any channel multipliers apply
            - channel_mult (tuple of ints): multipliers for number of inner channels to use
            - num_res_blocks (int): number of ResidualBlocks to use for each channel multiplier level
            - resblock_updown (bool): whether to use ResidualBlocks or Upsample/Downsample modules to resample
            - conv_resample (bool): if resblock_updown is False, whether to use Conv2D layers to resample
            - attention_resolutions (tuple of ints): which resolutions at which to apply attention
            - num_classes (int): if not None, number of classes to use for class-conditional models
            - num_heads (int): number of heads to use for AttentionBlocks
            - num_head_channels (int): number of channels to use for each head for AttentionBlocks, supersedes num_heads
            - use_scale_shift_norm (bool): whether to use scale and shift norm with step embeddings in ResidualBlocks
            - dropout (double): dropout probability in the ResidualBlocks

        Returns:
            - A UNet model to be used for diffusion.
    """
    def __init__(
            self,
            resolution,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            num_classes=None,
            num_heads=1,
            num_head_channels=None,
            resblock_updown=False,
            use_scale_shift_norm=False
    ):
        super(DiffusionModel, self).__init__()
        self.resolution = resolution
        self.in_channels = in_channels

        # Create embedding pipeline for time steps
        step_embed_dim = 4 * model_channels
        self.model_channels = model_channels
        self.step_embed = nn.Sequential(
            nn.Linear(model_channels, step_embed_dim),
            nn.SiLU(),
            nn.Linear(step_embed_dim, step_embed_dim)
        )

        # Create embedding for classes
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, embedding_dim=step_embed_dim)
            self.conditional = True
        else:
            self.conditional = False

        # Downsampling blocks
        self._feature_size = curr_channels = input_channels = int(model_channels * channel_mult[0])
        curr_res = resolution

        self.downsampling = nn.ModuleList([UsesStepsSequential(torch.nn.Conv2d(in_channels, curr_channels,
                                                                               kernel_size=(3, 3), stride=(1, 1),
                                                                               padding=(1, 1)))])
        input_block_channels = [curr_channels]  # Keep track of how many channels each block operates with
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # num_res_blocks residual blocks per level, plus an attention layer if specified
                layers = [ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim, dropout=dropout,
                                        out_channels=int(model_channels * mult),
                                        use_scale_shift_norm=use_scale_shift_norm)]
                curr_channels = int(model_channels * mult)

                # Add attention layer if specified to be added at this downsampling
                if curr_res in attention_resolutions:
                    layers.append(AttentionBlock(channels=curr_channels,
                                                 num_heads=num_heads, num_head_channels=num_head_channels))
                input_block_channels.append(curr_channels)
                self.downsampling.append(UsesStepsSequential(*layers))
                self._feature_size += curr_channels

            curr_channels = int(model_channels * mult)
            # Downsample at levels where the channel is multiplied
            if level != len(channel_mult) - 1:
                output_channels = curr_channels
                if resblock_updown:
                    self.downsampling.append(UsesStepsSequential(
                        ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim, dropout=dropout,
                                      out_channels=output_channels, downsample=True,
                                      use_scale_shift_norm=use_scale_shift_norm)))
                else:
                    self.downsampling.append(UsesStepsSequential(
                        Downsample(in_channels=curr_channels, out_channels=output_channels, with_conv=conv_resample)))
                curr_channels = output_channels
                input_block_channels.append(curr_channels)
                curr_res //= 2
                self._feature_size += curr_channels

        # Middle blocks - residual, attention, residual
        layers = [ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim,
                                dropout=dropout, use_scale_shift_norm=use_scale_shift_norm),
                  AttentionBlock(channels=curr_channels,
                                 num_heads=num_heads, num_head_channels=num_head_channels),
                  ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim, dropout=dropout,
                                use_scale_shift_norm=use_scale_shift_norm)]
        self.middle_block = UsesStepsSequential(*layers)
        self._feature_size += curr_channels

        # Upsampling blocks (pretty much reverse of downsampling blocks, also includes skip connections from before
        # various downsampling layers)
        self.upsampling = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [ResidualBlock(in_channels=curr_channels + skip_channels,  # append for skip connections
                                        step_channels=step_embed_dim, use_scale_shift_norm=use_scale_shift_norm,
                                        dropout=dropout, out_channels=int(model_channels * mult))]
                curr_channels = int(model_channels * mult)
                if curr_res in attention_resolutions:
                    layers.append(AttentionBlock(channels=curr_channels,
                                                 num_heads=num_heads, num_head_channels=num_head_channels))

                # Upsample at each channel multiplier layer
                if level != 0 and i == num_res_blocks:
                    output_channels = curr_channels
                    if resblock_updown:
                        layers.append(ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim,
                                                    dropout=dropout, out_channels=output_channels,
                                                    upsample=True, use_scale_shift_norm=use_scale_shift_norm))
                    else:
                        layers.append(Upsample(in_channels=curr_channels, out_channels=output_channels,
                                               with_conv=conv_resample))
                    curr_res *= 2

                self._feature_size += curr_channels
                self.upsampling.append(UsesStepsSequential(*layers))

        # Output
        self.out = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=curr_channels),
                                 nn.SiLU(),
                                 zero_module(nn.Conv2d(in_channels=input_channels, out_channels=out_channels,
                                                       kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

    def forward(self, x, timestep, y=None):
        assert (y is not None) == self.conditional, 'don\'t give y unless class-conditional model'
        assert x.shape[2] == self.resolution and x.shape[3] == self.resolution, \
            'incorrect resolution: {}'.format(x.shape[2:])

        embedding = self.step_embed(timestep_embedding(timestep, self.model_channels))

        if self.conditional:
            embedding += self.class_embedding(y)

        # input and downsampling
        xs = []
        for module in self.downsampling:
            x = module(x, embedding)
            xs.append(x)

        # middle block (res, attn, res)
        x = self.middle_block(x, embedding)
        # output and upsampling
        for module in self.upsampling:
            # Concatenate for skip connections from before the downsampling layers
            if not isinstance(module, Upsample):
                x = torch.cat([x, xs.pop()], dim=1)
            x = module(x, embedding)
        return self.out(x)


# ---------------------------------------------------------------------------------------------------------------------
# HELPER METHODS FOR MODEL IMPLEMENTATION
# ---------------------------------------------------------------------------------------------------------------------

# Helpful method that zeros all the parameters in a nn.Module, used for initialization
def zero_module(module):
    for param in module.parameters():
        param.detach().zero_()
    return module


# Method to create sinusoidal timestep embeddings, much like positional encodings found in Transformers
def timestep_embedding(timesteps, embedding_dim, max_period=10000):
    half = embedding_dim // 2
    emb = math.log(max_period) / half
    emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps[:, None].float() * emb[None]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
    # Zero pad for odd dimensions
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from collections import OrderedDict
from diffusion import Diffusion


# Show image (img must be RGB and from [0.0, 1.0] or [0, 255])
def imshow(img, title=None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# convert state dict from the ones from guided_diffusion to one compatible with my model, doesn't change sd
def convert_state_dict(sd):
    def convert_param_name(name):
        name = name.replace('input_blocks', 'downsampling')
        name = name.replace('output_blocks', 'upsampling')
        name = name.replace('in_layers.0', 'in_norm')
        name = name.replace('in_layers.2', 'in_conv')
        name = name.replace('emb_layers.1', 'step_embedding')
        name = name.replace('out_layers.0', 'out_norm')
        name = name.replace('out_layers.3', 'out_conv')
        name = name.replace('skip_connection', 'skip')
        name = name.replace('time_embed', 'step_embed')
        name = name.replace('qkv', 'qkv_nin')
        name = name.replace('label_emb', 'class_embedding')
        return name

    new_sd = OrderedDict()
    for _ in range(len(sd)):
        key, val = sd.popitem(False)
        old_key = key
        key = convert_param_name(key)
        sd[old_key] = val
        new_sd[key] = val

    return new_sd


# ---------------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# device = torch.device('cpu')
# torch.manual_seed(0)

STATE_DICT_FILENAME = 'models/64x64_diffusion.pt'
DIFFUSION_ARGS = {'rescaled_num_steps': 25, 'original_num_steps': 1000, 'use_ddim': True, 'ddim_eta': 1.0}
BATCH_SIZE = 1
NUM_SAMPLES = 1
SHOW_PROGRESS = True
# ---------------------------------------------------------------------------------------------------------------------

# CREATE MODEL
if STATE_DICT_FILENAME == 'models/64x64_diffusion.pt':
    CONDITIONAL = True
    DIFF_ARGS = {'beta_schedule': 'cosine', 'sampling_var_type': 'learned_range', 'device': device}
    MODEL_ARGS = {'resolution': 64, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 2, 3, 4),
                  'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 192,
                  'num_res_blocks': 3,
                  'resblock_updown': True, 'use_scale_shift_norm': True, 'num_classes': 1000 if CONDITIONAL else None}
elif STATE_DICT_FILENAME == 'diff256.pt':
    CONDITIONAL = False
    DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'learned_range', 'device': device}
    MODEL_ARGS = {'resolution': 256, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 2, 4, 4),
                  'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
                  'num_res_blocks': 2,
                  'resblock_updown': True, 'use_scale_shift_norm': True, 'num_classes': 1000 if CONDITIONAL else None}
elif STATE_DICT_FILENAME == 'cifar.pt':
    CONDITIONAL = False
    DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'small', 'device': device}
    MODEL_ARGS = {'resolution': 32, 'attention_resolutions': (16,), 'channel_mult': (1, 2, 2, 2),
                  'num_heads': 1, 'in_channels': 3, 'out_channels': 3, 'model_channels': 128,
                  'num_res_blocks': 2,
                  'resblock_updown': False, 'use_scale_shift_norm': False, 'num_classes': 1000 if CONDITIONAL else None}
else:
    raise NotImplementedError(STATE_DICT_FILENAME)

model = DiffusionModel(**MODEL_ARGS)
model.load_state_dict(convert_state_dict(torch.load(STATE_DICT_FILENAME, map_location="cpu")), strict=True)
model.to(device).eval()

print('Model made from {} with {} parameters! :)\n'.
      format(STATE_DICT_FILENAME, sum(p.numel() for p in model.parameters())))

print('Starting Diffusion! There are {} samples of {} images each\n'.format(NUM_SAMPLES, BATCH_SIZE))
samples = []
DIFFUSION_ARGS.update(DIFF_ARGS)
diffusion = Diffusion(model=model, **DIFFUSION_ARGS)
for i_sample in range(1, NUM_SAMPLES + 1):
    # CREATE RANDOM DATA
    data = torch.randn([BATCH_SIZE, 3, MODEL_ARGS['resolution'], MODEL_ARGS['resolution']]).to(device)
    labels = torch.randint(low=0, high=1000, size=(BATCH_SIZE,), device=device) if CONDITIONAL else None

    # RUN DIFFUSION
    print('Denoising sample {}! :)'.format(i_sample))
    out = diffusion.denoise(x=data, label=labels, batch_size=BATCH_SIZE, progress=SHOW_PROGRESS)

    # Convert from [-1.0, 1.0] to [0, 255]
    out = ((out + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    data = data.cpu().permute(0, 2, 3, 1).detach().numpy()
    out = out.cpu().detach().numpy()

    samples.append((data, out))
    print()

print('Displaying {} generated images!'.format(NUM_SAMPLES * BATCH_SIZE))
for sample in samples:
    data, out = sample
    for b in range(BATCH_SIZE):
        plt.close('all')
        fig = plt.figure(figsize=(7, 3))
        fig.add_subplot(1, 2, 1)
        imshow(data[b], title='Input Noise')
        fig.add_subplot(1, 2, 2)
        imshow(out[b], title='Output Image')
        plt.waitforbuttonpress()
