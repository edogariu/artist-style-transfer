import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

from diff_model import DiffusionModel
from diffusion import Diffusion
from classifier import ArtistClassifier

# ---------------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# device = torch.device('cpu')
torch.manual_seed(0)

STATE_DICT_FILENAME = 'models/64x64_diffusion.pt'
DIFFUSION_ARGS = {'rescaled_num_steps': 25, 'original_num_steps': 1000, 'use_ddim': True, 'ddim_eta': 0.0}

BATCH_SIZE = 4
NUM_SAMPLES = 1
DESIRED_LABELS = [300]  # set to list of labels (one for each sample) or [] for random label each sample

SHOW_PROGRESS = True
GUIDANCE = None  # can be None, 'classifier', or 'classifier_free'
GUIDANCE_STRENGTH = 1.0 if GUIDANCE is not None else None

# ---------------------------------------------------------------------------------------------------------------------

# CREATE MODEL(S)
if GUIDANCE == 'classifier':
    classifier = ArtistClassifier(state_dict_filename='models/best-2.pth', num_classes=19, device=device)
else:
    classifier = None

if STATE_DICT_FILENAME == 'models/64x64_diffusion.pt':
    CONDITIONAL = True
    DIFF_ARGS = {'beta_schedule': 'cosine', 'sampling_var_type': 'learned_range', 'classifier': classifier,
                 'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
                 'device': device}
    MODEL_ARGS = {'resolution': 64, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 2, 3, 4),
                  'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 192,
                  'num_res_blocks': 3,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000 if CONDITIONAL else None}
elif STATE_DICT_FILENAME == 'diff256.pt':
    CONDITIONAL = False
    DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'learned_range', 'classifier': classifier,
                 'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
                 'device': device}
    MODEL_ARGS = {'resolution': 256, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 2, 4, 4),
                  'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
                  'num_res_blocks': 2,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000 if CONDITIONAL else None}
elif STATE_DICT_FILENAME == 'cifar.pt':
    CONDITIONAL = False
    DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'small', 'classifier': classifier,
                 'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
                 'device': device}
    MODEL_ARGS = {'resolution': 32, 'attention_resolutions': (16,), 'channel_mult': (1, 2, 2, 2),
                  'num_heads': 1, 'in_channels': 3, 'out_channels': 3, 'model_channels': 128,
                  'num_res_blocks': 2,
                  'resblock_updown': False, 'use_adaptive_gn': False, 'num_classes': 1000 if CONDITIONAL else None}
else:
    raise NotImplementedError(STATE_DICT_FILENAME)


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


model = DiffusionModel(**MODEL_ARGS)
model.load_state_dict(convert_state_dict(torch.load(STATE_DICT_FILENAME, map_location="cpu")), strict=True)
model.to(device).eval()

print('Model made from {} with {} parameters! :)\n'.
      format(STATE_DICT_FILENAME, sum(p.numel() for p in model.parameters())))

print('Starting Diffusion! There are {} samples of {} images each\n'.format(NUM_SAMPLES, BATCH_SIZE))
samples = []
DIFFUSION_ARGS.update(DIFF_ARGS)
diffusion = Diffusion(model=model, **DIFFUSION_ARGS)
if CONDITIONAL and len(DESIRED_LABELS) != 0:
    assert len(DESIRED_LABELS) == NUM_SAMPLES, 'please provide NUM_SAMPLES={} labels'.format(NUM_SAMPLES)
for i_sample in range(NUM_SAMPLES):
    # CREATE RANDOM DATA
    data = torch.randn([BATCH_SIZE, 3, MODEL_ARGS['resolution'], MODEL_ARGS['resolution']]).to(device)
    if CONDITIONAL:
        if len(DESIRED_LABELS) == 0:
            labels = torch.randint(low=0, high=1000, size=(BATCH_SIZE,), device=device)
        else:
            labels = torch.full(size=(BATCH_SIZE,), fill_value=DESIRED_LABELS[i_sample], device=device)
    else:
        labels = None

    # RUN DIFFUSION
    print('Denoising sample {}! :)'.format(i_sample + 1))
    out = diffusion.denoise(x=data, kwargs={'y': labels}, batch_size=BATCH_SIZE, progress=SHOW_PROGRESS)

    # Convert from [-1.0, 1.0] to [0, 255]
    out = ((out + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    data = data.cpu().permute(0, 2, 3, 1).detach().numpy()
    out = out.cpu().detach().numpy()

    samples.append((data, out))
    print()


# Show image (img must be RGB and from [0.0, 1.0] or [0, 255])
def imshow(img, title=None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


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
