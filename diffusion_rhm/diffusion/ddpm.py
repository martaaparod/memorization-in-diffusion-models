from typing import Dict, Tuple


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DDPM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        model_type: str = "noise",
        model_output: str = "probabilities",
    ) -> None:
        super(DDPM, self).__init__()
        self.model = model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.model_type = model_type
        self.criterion = nn.MSELoss()
        self.readout = lambda x: x

        if model_output == "logits":
            self.readout = nn.Softmax(dim=1)
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, n_trajectories: int = 1) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using model.
        This implements Algorithm 1 in the paper.
        """
        if n_trajectories > 1:
            x_last_shape = x.shape[1], x.shape[2]
            x = (
                x.unsqueeze(1)
                .repeat(1, n_trajectories, 1, 1)
                .view(-1, x_last_shape[0], x_last_shape[1])
            )
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        eps = torch.randn_like(
            x, memory_format=torch.contiguous_format
        )  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        if self.model_type == "noise":
            return self.criterion(self.model(x_t, _ts / self.n_T), eps)
        elif self.model_type == "start":
            return self.criterion(self.model(x_t, _ts / self.n_T), x)
        elif self.model_type == "exact_score":
            score = torch.softmax(x * self.mean_over_var[_ts, None, None], 1)
            return self.criterion(self.model(x_t, _ts / self.n_T), score)

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.model(
                x_i,
                torch.tensor(i / self.n_T)
                .to(device)
                .repeat(
                    n_sample,
                ),
            )
            if self.model_type == "noise":
                eps = self.model(
                    x_i,
                    torch.tensor(i / self.n_T)
                    .to(device)
                    .repeat(
                        n_sample,
                    ),
                )
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
            elif self.model_type == "start" or self.model_type == "exact_score":

                x0 = self.readout(
                    self.model(
                        x_i,
                        torch.tensor(i / self.n_T)
                        .to(device)
                        .repeat(
                            n_sample,
                        ),
                    )
                )
                eps = (x_i - x0 * self.sqrtab[i]) / self.sqrtmab[
                    i
                ]  # eps = (x_i - x0 * sqrt(ab)) / sqrt(1-ab)
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

        return x_i


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    mean_over_var = torch.sqrt(alphabar_t) / (1 - alphabar_t)

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "mean_over_var": mean_over_var,  # \sqrt{\bar{\alpha_t}}/(1-\bar{\alpha_t)}
    }


class DiscreteDDPM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        model_type: str = "start",
        model_output: str = "logits",
    ) -> None:
        super(DiscreteDDPM, self).__init__()
        self.model = model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.model_type = model_type

        if model_output == "logits":
            self.readout = nn.Softmax(dim=1)
            # self.criterion = nn.CrossEntropyLoss()
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        elif model_output == "probabilities":
            self.readout = lambda x: x
            self.criterion = nn.MSELoss()

    def proba_posterior_0_t(
        self, xt: torch.Tensor, t_index: torch.Tensor):
        """
        Returns the posterior probability p(x_{0} | x_t, t).
        """
        v = xt.shape[1]
        p_x0_given_xt = self.alphabar_t[t_index, None, None] * xt + (
            1 - self.alphabar_t[t_index, None, None]
        ) / v * torch.ones_like(xt)
        return p_x0_given_xt # (B,v,d)

    def proba_posterior_t_1(
        self, xt: torch.Tensor, t_index: torch.Tensor, x0: torch.Tensor
    ):
        """
        Returns the posterior probability p(x_{t-1} | x_t, x_0, t).
        """
        v = xt.shape[1]
        proba1 = self.alpha_t[t_index, None, None] * xt + (
            1 - self.alpha_t[t_index, None, None]
        ) / v * torch.ones_like(xt)  # p(x_{t} | x_{t-1})
        proba2 = self.alphabar_t[t_index - 1, None, None] * x0 + (
            1 - self.alphabar_t[t_index - 1, None, None]
        ) / v * torch.ones_like(x0)  # p(x_{t-1} | x_0)
        proba = proba1 * proba2  # (B,v,d)
        return proba

    def noising_to_t(self, x: torch.Tensor, t_index: int, n_trajectories: int =1):
        """
        Sample x_t from q(x_t | x_0 = x, t = t_index).
        """
        v, d = x.shape[1], x.shape[2]

        if n_trajectories > 1:
            x = x.unsqueeze(1).repeat(1, n_trajectories, 1, 1).view(-1, v, d)

        # batch of timesteps
        _ts = torch.full((x.shape[0],), t_index, device=x.device, dtype=torch.long)

        proba = self.alphabar_t[_ts, None, None] * x + (
                1 - self.alphabar_t[_ts, None, None]
        ) / v * torch.ones_like(x)
        proba = proba.permute(0, 2, 1).reshape(-1, v)  # (B,v,d)->(B*d,v)

        x_t = torch.multinomial(proba, num_samples=1)  # (B*d, 1)
        x_t = nn.functional.one_hot(x_t, v).squeeze(-2)  # (B*d, v)
        x_t = x_t.reshape(-1, d, v).permute(0, 2, 1)  # (B,v,d)

        return x_t.float()

    def u_turn_sample(self, x: torch.Tensor, t_index: int, n_trajectories: int = 1):
        """
        Given a set of datapoints, adds noise to step t, and denoises them to get the denoised versions back
        """
        device = x.device
        v, d = x.shape[1], x.shape[2]

        # Forward (many independent noisy versions per sample)
        x_i = self.noising_to_t(x, t_index, n_trajectories)  # (B * n_trajectories, v, d)
        B = x_i.shape[0]

        # Reverse diffusion
        for i in range(t_index, 0, -1):
            # timestep input to the model, normalized
            t_norm = torch.full((B,), i / self.n_T, device=device)
            x0 = self.readout(self.model(x_i, t_norm))  # (B, v, d)

            # posterior p(x_{t-1} | x_t, x_0, t)
            t_idx = torch.full((B,), i, device=device, dtype=torch.long)
            proba = self.proba_posterior_t_1(x_i, t_idx, x0)  # (B, v, d)

            proba = proba.permute(0, 2, 1).reshape(-1, v)  # (B, v, d) -> (B*d, v)
            x_i = torch.multinomial(proba, num_samples=1)  # (B*d, 1)
            x_i = nn.functional.one_hot(x_i, num_classes=v).squeeze(-2)  # (B*d, v)
            x_i = x_i.reshape(-1, d, v).permute(0, 2, 1)  # (B, v, d)

        return x_i.float()

    def forward(self, x: torch.Tensor, n_trajectories: int = 1) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using model.
        This implements Algorithm 1 in the paper.
        """
        v, d = x.shape[1], x.shape[2]
        if n_trajectories > 1:
            x = x.unsqueeze(1).repeat(1, n_trajectories, 1, 1).view(-1, v, d)
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)

        proba = self.alphabar_t[_ts, None, None] * x + (
            1 - self.alphabar_t[_ts, None, None]
        ) / v * torch.ones_like(x)
        proba = proba.permute(0, 2, 1).reshape(-1, v)  # (B,v,d)->(B*d,v)

        x_t = torch.multinomial(proba, num_samples=1)  # (B*d,)
        x_t = nn.functional.one_hot(x_t, v)  # (B*d, v)
        x_t = x_t.reshape(-1, d, v).permute(0, 2, 1)  # (B*d,v)->(B,v,d)

        if self.model_type == "start":

            true_p = self.proba_posterior_t_1(x_t, _ts, x)
            true_p = true_p.permute(0, 2, 1).reshape(
                -1, true_p.shape[1]
            )  # (B,v,d)->(B*d,v)
            true_p = true_p / true_p.sum(1, keepdim=True)  # normalize
            model_p = self.proba_posterior_t_1(
                x_t, _ts, self.readout(self.model(x_t, _ts / self.n_T))
            )
            model_p = model_p.permute(0, 2, 1).reshape(
                -1, model_p.shape[1]
            )  # (B,v,d)->(B*d,v)
            model_p = model_p / model_p.sum(1, keepdim=True)  # normalize
            log_model_p = torch.log(model_p + 1e-8)  # to avoid log(0)
            log_true_p = torch.log(true_p + 1e-8)  # to avoid log(0)

            return (true_p * (log_true_p - log_model_p)).sum() / model_p.shape[
                0
            ]  # KLDivLoss

        else:

            raise NotImplementedError

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        v, d = size

        uniform_proba = torch.ones(size).to(device) / v  # (v,d)
        x_i = torch.multinomial(
            uniform_proba.T, num_samples=n_sample, replacement=True
        ).to(device)
        x_i = nn.functional.one_hot(x_i, v)  # (d,n)->(d,n,v)
        x_i = x_i.permute(1, 2, 0)  # (n,v,d)

        for i in range(self.n_T, 0, -1):

            x0 = self.readout(
                self.model(
                    x_i,
                    torch.tensor(i / self.n_T)
                    .to(device)
                    .repeat(
                        n_sample,
                    ),
                )
            )

            proba = self.proba_posterior_t_1(x_i, i, x0)  # (B,v,d)

            proba = proba.permute(0, 2, 1).reshape(-1, v)  # (B,v,d)->(B*d,v)
            x_i = torch.multinomial(proba, num_samples=1)  # (B*d,)
            x_i = nn.functional.one_hot(x_i, num_classes=v)
            x_i = x_i.reshape(-1, d, v).permute(0, 2, 1)

        return x_i.float()
    