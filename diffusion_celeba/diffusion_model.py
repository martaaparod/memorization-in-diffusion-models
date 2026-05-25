import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm
import torch.nn.functional as F
from unet import build_model

class Diffusion(nn.Module):
    def __init__(self, timesteps=1000, model_size='large', device="cuda"):
        super().__init__()
        self.device=device
        self.T = timesteps

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")
        self.noise_scheduler_sample = DDIMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")
        self.noise_scheduler_sample.set_timesteps(int(timesteps / 10)) # only used during sampling
        self.unet = build_model(model_size).to(device)

    def get_loss(self, x, t):
        # add noise
        noise = torch.randn_like(x)
        noisy_x = self.noise_scheduler.add_noise(x, noise, t)

        # get model prediction
        noise_pred = self.unet(noisy_x, t).sample

        # return loss
        return F.mse_loss(noise, noise_pred)

    def sample(self, batch_size):
        # set U-Net to evaluation mode
        self.unet.eval()

        with torch.no_grad():
            # get random noise
            x = torch.randn(batch_size, 3, 64, 64).to(self.device)

            # sampling loop
            for i, t in tqdm(enumerate(self.noise_scheduler_sample.timesteps)):
                # get model prediction
                residual = self.unet(x, t.to(self.device)).sample

                # update sample with step
                x = self.noise_scheduler_sample.step(residual, t, x).prev_sample

            # normalise to [0, 1] and move to cpu
            x = (x.detach().cpu().clip(-1, 1) + 1) / 2

        return x



