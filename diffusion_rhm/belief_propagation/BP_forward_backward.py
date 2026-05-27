import torch
from torch import nn
import time


def unif_sampling(x0, noise_lev, seed=1):
    torch.manual_seed(seed)
    B, v, d = x0.shape
    proba = (1 - noise_lev) * x0 + (noise_lev) / v * torch.ones_like(x0)
    proba = proba.permute(0, 2, 1).reshape(-1, v)  # (B,v,d)->(B*d,v)

    x_t = torch.multinomial(
        proba,
        num_samples=1,
    ).to(
        x0.device
    )  # (B*d,)
    x_t = nn.functional.one_hot(x_t, v)  # (B*d, v)
    x_t = x_t.reshape(-1, d, v).permute(0, 2, 1)  # (B*d,v)->(B,v,d)
    return x_t


def epsilon_process(x0, noise_level, bp, seed_forward=1, seed_backward=-1):
    B, v, d = x0.shape
    # xt = unif_sampling(x0, noise_level, seed_forward)

    x_leaves = x0.argmax(dim=1)
    # x_leaves = xt.argmax(dim=1)
    x_leaves = x_leaves.flatten()

    nu_up_L = bp.set_evidence_to_leaf_messages(x_leaves, noise=noise_level)
    new_x = bp.sample_from_upward_messages(nu_up_L, seed=seed_backward)
    new_x = new_x.reshape(B, d)
    new_x = nn.functional.one_hot(new_x, v).permute(0, 2, 1).float()

    return new_x


def masking(x0, noise_lev, seed=1):
    torch.manual_seed(seed)
    B, v, d = x0.shape
    proba = noise_lev * torch.ones(B*d)
    masking = torch.bernoulli(proba).to(x0.device) # (B*d,)

    return masking.reshape(-1, d)


def masking_process(x0, noise_level, bp, seed_forward=1, seed_backward=-1, return_masking=False):
    B, v, d = x0.shape
    masked = masking(x0, noise_level, seed_forward)

    x_leaves = x0.argmax(dim=1)
    x_leaves = x_leaves.flatten()
    masked_x = masked.flatten()
    masked_x = torch.nonzero(masked_x).flatten()

    nu_up_L = bp.set_masking_to_leaf_messages(x_leaves, masked_x)

    new_x = bp.sample_from_upward_messages(nu_up_L, seed=seed_backward)
    new_x = new_x.reshape(B, d)
    new_x = nn.functional.one_hot(new_x, v).permute(0, 2, 1).float()

    if return_masking:
        return new_x, masked
    else:
        return new_x


def backward_dynamics_masking_process(x0, noise_level, bp, seed_forward=1, seed_backward=-1, return_masking=False):
    time_start = time.time()
    
    dev = x0.device
    g = torch.Generator(device=dev)
    g.manual_seed(seed_backward)

    B, v, d = x0.shape
    
    def _step_backward_(num_masked, masked, leaf_marginals, g):
        # Sample one of the elements of masked
        # print("masked", masked.shape)

        idx = torch.floor(torch.rand(num_masked.shape, device=dev) * num_masked).int()
        idx = torch.cat((torch.tensor([0], device=dev), num_masked[:-1])).cumsum(dim=0) + idx
        idx = idx[num_masked>0].int()
        unmasked_idx = masked[idx]

        leaf_marginals = leaf_marginals[:, unmasked_idx]
        unmasked_token = torch.multinomial(leaf_marginals.T, num_samples=1, generator=g).flatten()
        
        # Remove all the unmasked tokens from masked through the index
        rem_mask = torch.ones_like(masked, device=dev)
        rem_mask[idx] = 0
        masked = masked[rem_mask.bool()]

        
        return unmasked_token, unmasked_idx, masked, g


    # Forward
    masked_batch = masking(x0, noise_level, seed_forward)

    x_leaves = x0.argmax(dim=1)
    x_leaves = x_leaves.flatten()
    new_x = x_leaves.clone()
    num_masked = masked_batch.sum(dim=1)
    masked_x = masked_batch.flatten()
    masked_x = torch.nonzero(masked_x).flatten()

    # print(x_leaves.device, masked_x.device, num_masked.device)

    num_steps = max(num_masked).int().item()

    for i in range(num_steps, 0, -1):
        if i%100==0:
            print(f"step {i}", flush=True)

        nu_up_L = bp.set_masking_to_leaf_messages(new_x, masked_x)
        leaf_marginals = bp.run_BP_from_upward_messages(nu_up_L)[bp.L] # (v, B*d)

        # sample one of the masked variables
        unmasked_token, unmasked_idx, masked_x, g = _step_backward_(num_masked, masked_x, leaf_marginals, g)
        new_x[unmasked_idx] = unmasked_token

        num_masked -= 1
        num_masked[num_masked<0] = 0
    
    new_x = new_x.reshape(B, d)
    new_x = nn.functional.one_hot(new_x, v).permute(0, 2, 1).float()

    time_end = time.time()
    print(f"Time taken for one batch: {time_end-time_start} s", flush=True)
    if return_masking:
        return new_x, masked_batch
    else:
        return new_x