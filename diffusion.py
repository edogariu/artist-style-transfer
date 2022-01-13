import math
import torch
import numpy as np
import tqdm

'''
DIFFUSION AND DENOISING UTILITIES BASED ON THE FOLLOWING PAPERS AND CORRESPONDING WORK:
    - Ho et al. Denoising Diffusion Probabilistic Models (DDPM): https://arxiv.org/pdf/2006.11239.pdf
    - Song et al. Denoising Diffusion Implicit Models (DDIM): https://arxiv.org/pdf/2010.02502.pdf
    - Dhariwal/Nichol Improved Denoising Diffusion Probabilistic Models (IDDPM): https://arxiv.org/pdf/2102.09672.pdf
    - Dhariwal/Nichol Diffusion Model Beats GAN on Image Synthesis (OpenAI): https://arxiv.org/pdf/2105.05233.pdf
    - Ho/Salismans Classifier-Free Diffusion Guidance (CFDG): 
                    https://openreview.net/pdf/ea628d03c92a49b54bc2d757d209e024e7885980.pdf
                    
possible improvements: 
    - during sampling, denoise fully, then diffuse to step s, then denoise again starting from s
    - truncate if using a CDM setup
'''


# Returns schedule for desired noise variance at each timestep
# Methods are linear, constant (uses beta_0), cosine (doesnt use beta_0 nor beta_T)
def get_beta_schedule(schedule_method, beta_0, beta_T, num_steps):
    if schedule_method == 'linear':
        betas = np.linspace(beta_0, beta_T, num_steps, dtype=np.float64)
    elif schedule_method == 'constant':
        betas = beta_0 * np.ones(num_steps, dtype=np.float64)
    elif schedule_method == 'cosine':  # from (eq. 17 of IDDPM)
        # function f(t) described in (eq. 17 of IDDPM)
        def f(t):
            s = 0.008  # extra value to add to fraction to prevent singularity
            return math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2

        betas = []
        for step in range(num_steps):
            alphabar_t_minus_one = step / num_steps
            alphabar_t = (step + 1) / num_steps
            betas.append(min(1 - f(alphabar_t) / f(alphabar_t_minus_one), 0.999))  # clip beta to be <= 0.999
        return np.array(betas)
    else:
        raise NotImplementedError("unimplemented variance scheduling method: {}".format(schedule_method))
    return betas


# Index into a (numpy array)'s elements with index t (tensor), returning a tensor that is broadcastable to shape
def extract(a, t, broadcast_shape):
    result = torch.gather(torch.from_numpy(a).to(t.device).float(), 0, t.long())
    # Keep adding dimensions to results until right shape
    while len(result.shape) < len(broadcast_shape):
        result = result[..., None]
    return result.expand(broadcast_shape)


class Diffusion:
    """
    Creates an object to handle a diffusion chain and a reverse diffusion (denoising) chain, with or without DDIM
    sampling.

        Parameters:
            - model (DiffusionModel): trained model to predict epsilon from noisy image
            - original_num_steps (int): number of diffusion steps that model was trained with (T)
            - rescaled_num_steps (int): number of diffusion steps to be considered when sampling
            - sampling_var_type (str): type of variance calculation -- 'small' or 'large' for fixed variances of given
              sizes, 'learned' or 'learned_range' for variances predicted by model
            - beta_schedule (str): scheduling method for noise variances (betas) -- 'linear', 'constant', or 'cosine'
            - betas (np.array): alternative to beta_schedule where betas are directly supplied
            - guidance_method (str): method of denoising guidance to use -- None, 'classifier', or 'classifier_free'
            - classifier (nn.Module): if guidance_method == 'classifier', which classifier model to use
            - guidance_strength (double): if guidance_method is not None, controls strength of guidance method selected
            - use_ddim (bool): whether to use DDIM sampling and interpret rescaled_num_steps as number of DDIM steps
            - ddim_eta (double): value to be used when performing DDIM
            - device (torch.device): if not None, which device to perform diffusion with


        Returns:
            - Diffusion object to call .diffuse() or .denoise() with.
    """

    def __init__(self, model,
                 original_num_steps, rescaled_num_steps,
                 sampling_var_type,
                 betas=None, beta_schedule='linear',
                 guidance_method=None, guidance_strength=None, classifier=None,
                 use_ddim=False, ddim_eta=None, device=None):
        self.model = model
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.guidance = guidance_method
        if guidance_method != 'classifier' and guidance_method != 'classifier_free' and guidance_method is not None:
            raise NotImplementedError(guidance_method)
        assert guidance_method is None or self.model.conditional, 'can only use guidance if model is conditional'
        self.strength = guidance_strength
        self.classifier = classifier
        if self.classifier is not None:
            self.classifier.float().to(self.device)
            self.classifier.eval()

        self.original_num_steps = original_num_steps
        self.rescaled_num_steps = rescaled_num_steps
        self.sampling_var_type = sampling_var_type

        if use_ddim:
            assert ddim_eta is not None, 'please supply eta if you want to use ddim'
        self.use_ddim = use_ddim
        self.ddim_eta = ddim_eta

        if betas is None:
            betas = get_beta_schedule(beta_schedule, 0.0001, 0.02, original_num_steps)
        else:
            assert len(betas) == original_num_steps, 'betas must be the right length!'
            betas = np.array(betas, dtype=np.float64)

        # Rescale betas to match the number of rescaled diffusion steps with (eq. 19 in IDDPM)
        alphas = 1.0 - betas  # array of alpha_t for indices t
        alphas_cumprod = np.cumprod(alphas, axis=0)  # alphabar_t
        rescaled_timesteps = list(range(-20 + original_num_steps // (2 * rescaled_num_steps),
                                        original_num_steps + original_num_steps // (2 * rescaled_num_steps) - 20,
                                        original_num_steps // rescaled_num_steps))
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in rescaled_timesteps:
                new_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        betas = np.array(new_betas)

        self.betas = betas  # scheduled noise variance for each timestep
        self.timestep_map = torch.tensor(rescaled_timesteps,
                                         device=device, dtype=torch.long)  # map from rescaled to original timesteps

        # calculate and store various values to be used in diffusion, denoising, and ddim denoising
        # All these are arrays whose values at index t correspond to the comments next to them
        alphas = 1.0 - betas  # alpha_t
        sqrt_alphas = np.sqrt(alphas)  # sqrt(alpha_t)
        self.alphas_cumprod = alphas_cumprod = np.cumprod(alphas, axis=0)  # alphabar_t
        self.alphas_cumprod_prev = alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])  # alphabar_{t-1}
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)  # sqrt(alphabar_t)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)  # sqrt(1 - alphabar_t)
        self.sqrt_reciprocal_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)  # sqrt(1 / alphabar_t)
        self.sqrt_reciprocal_alphas_minus_one_cumprod = \
            np.sqrt(1.0 / alphas_cumprod - 1)  # sqrt((1 - alphabar_t) / alphabar_t)

        # Calculate posterior means and variances for forward (i.e. q sampling/diffusion) process, (eq. 7 in DDPM)
        self.posterior_mean_coef_x0 = np.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
        self.posterior_mean_coef_xt = sqrt_alphas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Clip to remove variance at 0, since it will be strange
        self.log_posterior_var_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))

        # DDPM implements fixed variances, but IDDPM prefer to learn the variances
        if sampling_var_type == 'large':
            self.log_var = np.log(np.append(posterior_variance[1], betas[1:]))
        elif sampling_var_type == 'small':
            self.log_var = np.log(np.maximum(posterior_variance, 1e-20))
        elif sampling_var_type == 'learned' or sampling_var_type == 'learned_range':
            self.log_var = None
        else:
            raise NotImplementedError(sampling_var_type)

    def diffuse(self, x, steps_to_do=None):
        """
        Add noise to an input corresponding to a given number of steps in the diffusion Markov chain.

            Parameters:
                - x (torch.tensor): input image to diffuse (usually x_0)
                - steps_to_do (int): number of rescaled diffusion steps to apply forward

            Returns:
                - Diffused image(s).
        """
        with torch.no_grad():
            # If unspecified or invalid number of steps to do, go until completion x_T
            if steps_to_do is None or steps_to_do > self.rescaled_num_steps:
                steps_to_do = self.rescaled_num_steps

            timestep = (steps_to_do * torch.ones(x.shape[0])).to(self.device)
            x = self.diffusion_step(x, t=timestep)
        return x

    def denoise(self, x=None, kwargs=None, start_step=None, steps_to_do=None, batch_size=1, progress=True):
        """
        Sample the posterior of the forward process in the diffusion Markov chain. If self.use_ddim is True, uses DDIM
        sampling instead of traditional DDPM sampling.

            Parameters:
                - x (torch.tensor): if not None, input image x_t to denoise. if None, denoises Gaussian noise x_T
                - kwargs (dict): dict of extra args to pass to model, should be {'y': label}  with label to guide with
                - start_step (int): which rescaled step to start at. should correspond to x's timestep
                - steps_to_do (int): number of rescaled diffusion steps to apply forward
                - batch_size (int): batch size of denoising process. only used if x is None
                - progress (bool): whether to show tqdm progress bar

            Returns:
                - Denoised sample(s).
        """
        if kwargs is None:
            kwargs = {}
        assert ('y' in kwargs.keys() and kwargs['y'] is not None) == self.model.conditional, \
            'pass label iff model is class-conditional'
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

            # Apply timestep rescaling
            indices = list(range(start_step - steps_to_do, start_step))
            # Add progress bar if needed
            if progress:
                progress_bar = tqdm.tqdm
                indices = progress_bar(reversed(indices), total=self.rescaled_num_steps)
            else:
                indices = list(reversed(indices))

            assert len(indices) == steps_to_do
            for t in indices:
                timestep = (t * torch.ones(x.shape[0])).to(self.device)
                if not self.use_ddim:  # NORMAL DDPM
                    x, x_0 = self.denoising_step(x, t=timestep, kwargs=kwargs)
                else:  # DDIM SAMPLING
                    x, x_0 = self.ddim_denoising_step(x, t=timestep, kwargs=kwargs)
        return x

    # -----------------------------------------------------------------------------------------------------------------

    # Samples from q(x_t | x_0), i.e. applies t steps of noise to x_0
    def diffusion_step(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        # (eq. 4 in DDPM paper)
        return extract(self.sqrt_alphas_cumprod, t, x.shape) * x + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise

    # Samples from p(x_{t-1} | x_t), i.e. use model to predict noise, then samples from corresponding possible x_{t-1}'s
    # If return_x0, also returns the predicted initial image x_0
    def denoising_step(self, x_t, t, kwargs=None, clip_x=True):
        eps_pred = self.model(x_t, self.timestep_map[t.long()], **kwargs)
        if self.sampling_var_type == 'learned':
            eps_pred, log_var = torch.split(eps_pred, int(eps_pred.shape[1] / 2), dim=1)
        elif self.sampling_var_type == 'learned_range':
            assert self.log_posterior_var_clipped is not None and self.betas is not None
            eps_pred, log_var = torch.split(eps_pred, int(eps_pred.shape[1] / 2), dim=1)

            min_log = extract(self.log_posterior_var_clipped, t, x_t.shape)
            max_log = extract(np.log(self.betas), t, x_t.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var]
            frac = (log_var + 1) / 2
            log_var = frac * max_log + (1 - frac) * min_log
        else:
            log_var = extract(self.log_var, t, x_t.shape)

        # If using classifier-free guidance, push eps_pred in the direction unique to its class and
        # away from the base prediction (eq. 6 in CFDG paper):
        # eps_pred(x_t, c) <- (1 + w) * eps_pred(x_t, c) - w * eps_pred(x_t, -1), where -1 is the null class
        if self.guidance == 'classifier_free':
            base_eps_pred = self.model(x_t, self.timestep_map[t.long()],
                                       y=torch.tensor([-1] * eps_pred.shape[0], device=self.device))
            if self.sampling_var_type == 'learned' or self.sampling_var_type == 'learned_range':
                base_eps_pred, _ = torch.split(base_eps_pred, int(base_eps_pred.shape[1] / 2), dim=1)
            eps_pred = (1 + self.strength) * eps_pred - self.strength * base_eps_pred

        # Predict x_start from x_t and epsilon (eq. 11 in DDPM paper)
        pred_x0 = extract(self.sqrt_reciprocal_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_reciprocal_alphas_minus_one_cumprod, t, eps_pred.shape) * eps_pred
        if clip_x:
            pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Calculate mean of posterior q(x_{t-1} | x_t, x_0) (eq. 7 in DDPM paper)
        mean = extract(self.posterior_mean_coef_x0, t, pred_x0.shape) * pred_x0 + extract(
            self.posterior_mean_coef_xt, t, x_t.shape) * x_t

        # If we use classifier guidance, add to the mean the value: s * grad_{x_t}[log(classifier prob)]
        # This is (Algorithm 1 in OpenAI paper)
        if self.guidance == 'classifier':
            with torch.enable_grad():
                x = x_t.detach().requires_grad_(True)
                classifier_log_probs = torch.log_softmax(self.classifier(x), dim=-1)  # ADD T AS INPUT FOR NOISY CLASSIFIER
                # Grab log probabilities of desired labels for each element of batch
                grabbed = classifier_log_probs[range(classifier_log_probs.shape[0]), torch.flatten(kwargs['y'])]
                grad = torch.autograd.grad(grabbed.sum(), x)[0]  # grad = grad_{x_t}[log(p[y | x_t, t])]
                mean += self.strength * grad * torch.exp(log_var)

        # Return sample pred for x_0 using calculated mean and given log variance, evaluated at desired timestep
        # (Between eq. 11 and eq. 12 in DDPM paper)
        noise = torch.randn_like(x_t)
        mask = 1.0 - (t == 0).float()
        mask = mask.reshape((x_t.shape[0],) + (1,) * (len(x_t.shape) - 1))

        sample = mean + mask * torch.exp(0.5 * log_var) * noise
        sample = sample.float()

        return sample, pred_x0

    # Implement denoising diffusion implicit models (DDIM): https://arxiv.org/pdf/2010.02502.pdf
    def ddim_denoising_step(self, x_t, t, kwargs=None, clip_x=True):
        eps_pred = self.model(x_t, self.timestep_map[t.long()], **kwargs)
        if self.sampling_var_type == 'learned' or self.sampling_var_type == 'learned_range':
            eps_pred, _ = torch.split(eps_pred, int(eps_pred.shape[1] / 2), dim=1)

        # If we use classifier guidance, subtract from eps_pred the value:
        # s * sqrt(1 - alpha_bar) grad_{x_t}[log(classifier prob)]
        # This is (Algorithm 2 in OpenAI paper)
        if self.guidance == 'classifier':
            with torch.enable_grad():
                x = x_t.detach().requires_grad_(True)
                classifier_log_probs = torch.log_softmax(self.classifier(x), dim=-1)  # ADD T AS INPUT FOR NOISY CLASSIFIER
                # Grab log probabilities of desired labels for each element of batch
                grabbed = classifier_log_probs[range(classifier_log_probs.shape[0]), torch.flatten(kwargs['y'])]
                grad = torch.autograd.grad(grabbed.sum(), x)[0]  # grad = grad_{x_t}[log(p[y | x_t, t])]
                eps_pred -= self.strength * grad * extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # If using classifier-free guidance, push eps_pred in the direction unique to its class and
        # away from the base prediction (eq. 6 in CFDG paper):
        # eps_pred(x_t, c) <- (1 + w) * eps_pred(x_t, c) - w * eps_pred(x_t, -1), where -1 is the null class
        elif self.guidance == 'classifier_free':
            base_eps_pred = self.model(x_t, self.timestep_map[t.long()],
                                       y=torch.tensor([-1] * eps_pred.shape[0], device=self.device))
            if self.sampling_var_type == 'learned' or self.sampling_var_type == 'learned_range':
                base_eps_pred, _ = torch.split(base_eps_pred, int(base_eps_pred.shape[1] / 2), dim=1)
            eps_pred = (1 + self.strength) * eps_pred - self.strength * base_eps_pred

        # Same as in DDPM (eq. 11)
        pred_x0 = extract(self.sqrt_reciprocal_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_reciprocal_alphas_minus_one_cumprod, t, x_t.shape) * eps_pred
        if clip_x:
            pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Accelerated sampling of Generative Process (secs. 4.1 and 4.2 in DDIM)
        # (eq. 12 in DDIM)
        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x_t.shape)
        var = self.ddim_eta ** 2 * (1.0 - alpha_bar_prev) * (1.0 - alpha_bar / alpha_bar_prev) / (1.0 - alpha_bar)
        mean = pred_x0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - var) * eps_pred

        noise = torch.randn_like(x_t)
        mask = 1.0 - (t == 0).float()
        mask = mask.reshape((x_t.shape[0],) + (1,) * (len(x_t.shape) - 1))

        sample = mean + mask * torch.sqrt(var) * noise
        sample = sample.float()

        return sample, pred_x0
