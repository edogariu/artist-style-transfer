import math
import torch
import numpy as np
import tqdm


# Returns schedule for desired noise variance at each timestep
# Methods are linear, constant (uses beta_0), cosine (doesnt use beta_0 nor beta_T)
def get_beta_schedule(schedule_method, beta_0, beta_T, num_steps):
    if schedule_method == 'linear':
        betas = np.linspace(beta_0, beta_T, num_steps, dtype=np.float64)
    elif schedule_method == 'constant':
        betas = beta_0 * np.ones(num_steps, dtype=np.float64)
    elif schedule_method == 'cosine':
        def alpha_mean_func(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for s in range(num_steps):
            s1 = s / num_steps
            s2 = (s + 1) / num_steps
            betas.append(min(1 - alpha_mean_func(s2) / alpha_mean_func(s1), 0.999))
        return np.array(betas)
    else:
        raise NotImplementedError(f"unimplemented variance scheduling method: {schedule_method}")
    return betas


# Extract a (numpy array)'s elements based on t (tensor) in a way that is broadcastable with shape
def extract(a, t, broadcast_shape):
    result = torch.gather(torch.from_numpy(a).to(t.device).float(), 0, t.long())
    # Keep adding dimensions to results
    while len(result.shape) < len(broadcast_shape):
        result = result[..., None]
    return result.expand(broadcast_shape)


# Samples from q(x_t | x_0), i.e. applies t steps of noise to x_0
def noising_step(x, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x)
    # eq. 2 in Ho et al. DDPM paper
    return extract(sqrt_alphas_cumprod, t, x.shape) * x + extract(sqrt_one_minus_alphas_cumprod, t, x.shape) * noise


# Samples from p(x_{t-1} | x_t), i.e. uses model to predict noise, then samples from corresponding possible x_{t-1}'s
# If return_x0, also returns the predicted initial image x_0
def denoising_step(x_t, t, timestep_map, model, log_var, var_type,
                   sqrt_reciprocal_alphas_cumprod, sqrt_reciprocal_alphas_minus_one_cumprod,
                   posterior_mean_coef1, posterior_mean_coef2,
                   log_postvar=None, betas=None, y=None, return_x0=False, clip_x=True):
    eps_pred = model(x_t, timestep_map[t.long()], y=y)
    if var_type == 'learned':
        c = int(eps_pred.shape[1] / 2)
        eps_pred, log_var = torch.split(eps_pred, c, dim=1)
    elif var_type == 'learned_range':
        assert log_postvar is not None and betas is not None
        c = int(eps_pred.shape[1] / 2)
        eps_pred, log_var = torch.split(eps_pred, c, dim=1)

        min_log = extract(log_postvar, t, x_t.shape)
        max_log = extract(np.log(betas), t, x_t.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var]
        frac = (log_var + 1) / 2
        log_var = frac * max_log + (1 - frac) * min_log
    else:
        log_var = extract(log_var, t, x_t.shape)
    pred_x0 = extract(sqrt_reciprocal_alphas_cumprod, t, x_t.shape) * x_t - \
              extract(sqrt_reciprocal_alphas_minus_one_cumprod, t, x_t.shape) * eps_pred
    if clip_x:
        pred_x0 = torch.clamp(pred_x0, -1, 1)

    # Calculate mean of q(x_{t-1} | x_t, x_0) (eq. 6 in Ho et al. DDPM paper)
    mean = extract(posterior_mean_coef1, t, x_t.shape) * pred_x0 + extract(posterior_mean_coef2, t, x_t.shape) * x_t

    # Return sample pred for x_0 using calculated mean and given log variance
    noise = torch.randn_like(x_t)
    mask = 1.0 - (t == 0).float()
    mask = mask.reshape((x_t.shape[0],) + (1,) * (len(x_t.shape) - 1))

    sample = mean + mask * torch.exp(0.5 * log_var) * noise
    sample = sample.float()

    if return_x0:
        return sample, pred_x0
    return sample


def ddim_denoising_step(x_t, t, timestep_map, model, eta, var_type, alphas_cumprod, alphas_cumprod_prev,
                        sqrt_reciprocal_alphas_cumprod, sqrt_reciprocal_alphas_minus_one_cumprod,
                        y=None, return_x0=False, clip_x=True):
    eps_pred = model(x_t, timestep_map[t.long()], y=y)
    if var_type == 'learned' or var_type == 'learned_range':
        c = int(eps_pred.shape[1] / 2)
        eps_pred, _ = torch.split(eps_pred, c, dim=1)

    pred_x0 = extract(sqrt_reciprocal_alphas_cumprod, t, x_t.shape) * x_t - \
              extract(sqrt_reciprocal_alphas_minus_one_cumprod, t, x_t.shape) * eps_pred
    if clip_x:
        pred_x0 = torch.clamp(pred_x0, -1, 1)

    eps_pred = (extract(sqrt_reciprocal_alphas_cumprod, t, x_t.shape) * x_t - pred_x0) / \
               extract(sqrt_reciprocal_alphas_minus_one_cumprod, t, x_t.shape)
    alpha_bar = extract(alphas_cumprod, t, x_t.shape)
    alpha_bar_prev = extract(alphas_cumprod_prev, t, x_t.shape)
    var = eta ** 2 * (1.0 - alpha_bar_prev) * (1.0 - alpha_bar / alpha_bar_prev) / (1.0 - alpha_bar)

    # equation 12 from openai
    mean = pred_x0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - var) * eps_pred

    noise = torch.randn_like(x_t)
    mask = 1.0 - (t == 0).float()
    mask = mask.reshape((x_t.shape[0],) + (1,) * (len(x_t.shape) - 1))

    sample = mean + mask * torch.sqrt(var) * noise
    sample = sample.float()

    if return_x0:
        return sample, pred_x0
    return sample


class Diffusion:
    def __init__(self, model, original_num_steps, rescaled_num_steps, sampling_var_type, betas=None,
                 beta_schedule='linear',
                 use_ddim=False, ddim_eta=None, device=None):
        self.model = model
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.model.to(self.device)
        self.original_num_steps = original_num_steps
        self.rescaled_num_steps = rescaled_num_steps
        self.sampling_var_type = sampling_var_type
        assert use_ddim == (ddim_eta is not None), 'please pick if u do or dont want ddim'
        self.use_ddim = use_ddim
        self.ddim_eta = ddim_eta

        if betas is None:
            betas = get_beta_schedule(beta_schedule, 0.0001, 0.02, original_num_steps)
        else:
            betas = np.array(betas, dtype=np.float64)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # Rescale betas
        rescaled_timesteps = list(range(3, original_num_steps + 3, original_num_steps // rescaled_num_steps))
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in rescaled_timesteps:
                new_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        betas = np.array(new_betas)
        alphas = 1.0 - betas
        sqrt_alphas = np.sqrt(alphas)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.betas = betas
        self.timestep_map = torch.tensor(rescaled_timesteps, device=device, dtype=torch.long)

        # values for ddim posterior
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_0) i.e. noising and others
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0) i.e. denoising
        self.sqrt_reciprocal_alphas_cumprod = np.sqrt(np.reciprocal(alphas_cumprod))
        self.sqrt_reciprocal_alphas_minus_one_cumprod = np.sqrt(np.reciprocal(alphas_cumprod) - 1)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)  # from sec 3.2 of Ho et. al
        self.log_postvar_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * sqrt_alphas / (1.0 - alphas_cumprod)
        # Ho et. al implement fixed variances, but Dhariwal/Nichol prefer to learn the variances
        if sampling_var_type == 'large':
            self.log_var = np.log(np.append(posterior_variance[1], betas[1:]))
        elif sampling_var_type == 'small':
            self.log_var = np.log(np.maximum(posterior_variance, 1e-20))
        elif sampling_var_type == 'learned' or sampling_var_type == 'learned_range':
            self.log_var = None
        else:
            raise NotImplementedError(sampling_var_type)

    def diffuse(self, x, steps_to_do=None, batch_size=1):
        with torch.no_grad():
            # If unspecified or invalid number of steps to do, go until completion x_T
            if steps_to_do is None or steps_to_do > self.original_num_steps:
                steps_to_do = self.original_num_steps

            timestep = (steps_to_do * torch.ones(batch_size)).to(self.device)
            x = noising_step(x, t=timestep,
                             sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                             sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)
        return x

    def denoise(self, x=None, label=None, start_step=None, steps_to_do=None, batch_size=1, progress=True):
        assert (label is not None) == self.model.conditional, 'pass label iff model is class-conditional'
        with torch.no_grad():
            # If no specified starting step, start from x = x_T
            if start_step is None:
                start_step = self.rescaled_num_steps

            # If unspecified or invalid number of steps to do, go until completion x_0
            if steps_to_do is None or steps_to_do > start_step:
                steps_to_do = start_step

            # If unspecified x, start with x_T that is noise
            if x is None:
                assert start_step == self.rescaled_num_steps, 'cannot start from noise with current step that is not T'
                x = torch.randn(batch_size, self.model.in_channels, self.model.resolution, self.model.resolution). \
                    to(self.device)

            # Apply timestep rescaling if needed
            if not self.use_ddim:
                indices = list(range(start_step - steps_to_do, start_step))
                assert len(indices) == steps_to_do
                # Add progress bar if needed
                if progress:
                    progress_bar = tqdm.tqdm
                    indices = progress_bar(reversed(indices), total=self.rescaled_num_steps)
                else:
                    indices = reversed(indices)
                for t in indices:
                    timestep = (t * torch.ones(batch_size)).to(self.device)
                    x, x_0 = denoising_step(x, t=timestep, timestep_map=self.timestep_map,
                                            y=label, model=self.model, return_x0=True,
                                            var_type=self.sampling_var_type, betas=self.betas,
                                            log_var=self.log_var, log_postvar=self.log_postvar_clipped,
                                            sqrt_reciprocal_alphas_cumprod=self.sqrt_reciprocal_alphas_cumprod,
                                            sqrt_reciprocal_alphas_minus_one_cumprod=self.
                                            sqrt_reciprocal_alphas_minus_one_cumprod,
                                            posterior_mean_coef1=self.posterior_mean_coef1,
                                            posterior_mean_coef2=self.posterior_mean_coef2)

            else:  # APPLY DDIM
                indices = list(range(start_step - steps_to_do, start_step))
                assert len(indices) == self.rescaled_num_steps
                # Add progress bar if needed
                if progress:
                    progress_bar = tqdm.tqdm
                    indices = progress_bar(reversed(indices), total=self.rescaled_num_steps)
                else:
                    indices = reversed(indices)
                for t in indices:
                    timestep = (t * torch.ones(batch_size)).to(self.device)
                    x, x_0 = ddim_denoising_step(x, t=timestep, timestep_map=self.timestep_map,
                                                 y=label, model=self.model, return_x0=True,
                                                 var_type=self.sampling_var_type, eta=self.ddim_eta,
                                                 alphas_cumprod=self.alphas_cumprod,
                                                 alphas_cumprod_prev=self.alphas_cumprod_prev,
                                                 sqrt_reciprocal_alphas_cumprod=self.sqrt_reciprocal_alphas_cumprod,
                                                 sqrt_reciprocal_alphas_minus_one_cumprod=self.
                                                 sqrt_reciprocal_alphas_minus_one_cumprod)
        return x
