import torch
import torch.nn as nn
import torch.optim as optim

from diff_model import DiffusionModel

# ADD SMARTER BETA SCHEDULE SAMPLING
# ADD DISTRIBUTED TRAINING WITH TORCH.DIST
# ADD 16 BIT PRECISION?
# ADD ANNEALED LR?

class BatchGenerator:
    def __init__(self, resolution, batch_size, conditional, random_crop=False, random_flip=True):
        self.batch_size = batch_size
        self.conditional = conditional

        # DO THIS LATER

class Trainer:
    def __init__(self, model, diffusion, dataloader,
                 iterations, batch_size, lr, weight_decay, ema_rates=(0.9999,), microbatch_size=None,
                 checkpoint=(None, None, None, None), save_every=None, device=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = model.to(device)
        self.device = device
        self.model.train()
        self.diffusion = diffusion

        self.loader = dataloader
        self.batch_size = microbatch_size if microbatch_size is not None else batch_size

        self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

        # checkpoint is tuple of state_dict path, ema_params path, and training step we are resuming at
        if any(c is not None for c in checkpoint):
            print('hasnone')
            assert not checkpoint.__contains__(None), \
                'please provide model, ema, and optimizer checkpoint paths and number of steps to resume with'
            self.curr_step = checkpoint[2]
            self.model.load_state_dict(torch.load(checkpoint[0], map_location='cpu'), strict=True)
            self.opt.load_state_dict(torch.load(checkpoint[1], map_location='cpu'))
            ema_state_dict = torch.load(checkpoint[1], map_location='cpu')
            self.ema = [ema_state_dict[name] for name, _ in model.named_parameters()]
        else:
            self.curr_step = 0
            self.ema = []  # ?
        self.iterations = iterations
        self.save_every = save_every


    def train(self):
        for step in range(self.iterations):
            batch, labels = next(self.loader)

            self.model.zero_grad()
            for i_micro in range(0, batch.shape[0], self.batch_size):
                micro = batch[i_micro: i_micro + self.batch_size].to(self.device)
                labels = {'y': labels[i_micro: i_micro + self.batch_size].to(self.device)}
                t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            self.forward_backward(batch, labels)
            took_step = self.mp_trainer.optimize(self.opt)
            if took_step:
                self._update_ema()
            self._anneal_lr()
            self.log_step()

            if self.save_every is not None and step % self.save_every == 0:
                self.save()

        return 0

    def training_step(self, batch, labels):
        return 0

    def save(self):



if __name__ == '__main__':
    MODEL_ARGS = {'resolution': 128, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 3, 4),
                  'num_heads': 4, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
                  'num_res_blocks': 2,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000}
    model = DiffusionModel(**MODEL_ARGS)
    dataloader = BatchGenerator(128, 8, True)
    trainer = Trainer(model=model, diffusion=None, dataloader=dataloader, batch_size=8, lr=0.001, weight_decay=0.999)