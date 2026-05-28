import torch
import torch.nn as nn
from belief_propagation.BP_utils import BP_countcorrect_upward

import numpy as np


def check_generated_tuples(rules_L, samples):
    """
    Check if the generated data is consistent with the rules.
    """
    # print(len(rules_L.reshape(-1, rules_L.shape[-1]))/8**2)
    samples = torch.argmax(samples, dim=1)
    tuple_size = rules_L.shape[-1]
    rules_L = rules_L.reshape(-1, rules_L.shape[-1])

    count_dict = {}
    for rule in rules_L:
        count_dict[tuple(rule.flatten().tolist())] = 0

    total_tuples = 0
    for tup in samples.view(-1, tuple_size):
        total_tuples += 1
        occ = (tup == rules_L.squeeze()).prod(1)
        if occ.sum() == 1:
            ir = torch.argmax(occ).item()
            count_dict[tuple(rules_L[ir].flatten().tolist())] += 1

    generation_acc = sum([i for i in count_dict.values()]) / total_tuples
    empirical_freqs = sorted(
        [(k, i / len(samples) * len(rules_L)) for k, i in count_dict.items()]
    )

    return generation_acc, empirical_freqs


def cross_entropy_tuples(rules_L, samples):
    """
    Check if the generated data is consistent with the rules.
    """
    # print(len(rules_L.reshape(-1, rules_L.shape[-1]))/8**2)
    probas = samples
    tuple_size = rules_L.shape[-1]
    rules_L = rules_L.reshape(-1, rules_L.shape[-1])
    cumul_proba = 0
    total_tuples = 0
    for tup in probas:
        total_tuples += 1
        rules_L = rules_L.squeeze()
        # print(tup.shape, rules_L.shape)
        # proba_rules = tup.gather(0,rules_L).prod(1).log().mean()
        # if total_tuples < 30: print(tup)
        proba_rules = tup.gather(0, rules_L).prod(1).mean()
        # print("logP", proba_rules)
        cumul_proba += proba_rules
        # if (tup == rules_L.squeeze()).prod(1).sum() == 1:
        #     correct_tuples += 1
    return cumul_proba / total_tuples


def check_generated_pixels(rules, samples):
    """
    Check if the generated data is consistent with the rules.
    """
    # print(samples)
    samples = torch.argmax(samples, dim=1)
    # print(samples)
    count_dict = {}
    for rule in rules.flatten():
        count_dict[rule.item()] = 0
    correct_samples = 0
    for sample in samples:
        if sample in rules.flatten():
            count_dict[sample.item()] += 1

    generation_acc = sum([i for i in count_dict.values()]) / len(samples)
    empirical_freqs = sorted(
        [(k, i / len(samples) * len(rules.flatten())) for k, i in count_dict.items()]
    )
    return generation_acc, empirical_freqs


def compute_loss_per_time(n_windows, points_per_window, model, x0):

    with torch.no_grad():
        model.eval()

        time_losses = {}
        for time_window in range(n_windows):
            n_trajectories = points_per_window
            x0_last_shape = x0.shape[1], x0.shape[2]
            x = (
                x0.unsqueeze(1)
                .repeat(1, n_trajectories, 1, 1)
                .view(-1, x0_last_shape[0], x0_last_shape[1])
                .to(x0.device)
            )
            _ts = torch.randint(
                model.n_T // n_windows * time_window,
                model.n_T // n_windows * (time_window + 1),
                (x.shape[0],),
            ).to(x.device)
            eps = torch.randn_like(x, memory_format=torch.contiguous_format)
            x_t = (
                model.sqrtab[_ts, None, None] * x + model.sqrtmab[_ts, None, None] * eps
            )

            if model.model_type == "noise":
                time_losses[time_window] = model.criterion(
                    model.model(x_t, _ts / model.n_T), eps
                ).item()
            elif model.model_type == "start":
                time_losses[time_window] = model.criterion(
                    model.model(x_t, _ts / model.n_T), x
                ).item()
            elif model.model_type == "exact_score":
                score = torch.softmax(x * model.mean_over_var[_ts, None, None], 1)
                time_losses[time_window] = (
                    model.criterion(model.model(x_t, _ts / model.n_T), score)
                ).item()
                if model.criterion == nn.CrossEntropyLoss:
                    entropy = -torch.sum(score * torch.log(score), dim=1).mean()
                    time_losses[time_window] -= entropy.item()

        return time_losses


def hierarhical_copies(samples, dataset, num_layers, tuple_size):
    """
    Check if substrings of the generated data are copies of the training set.
    """

    d = dataset.shape[2]
    # num_layers = 2
    # tuple_size = 2

    hierarhical_copies = {}
    for layer in range(1, num_layers+1):
        size_string = tuple_size ** (layer)
        num_strings = d // size_string

        sample_strings = samples.reshape(samples.shape[0], samples.shape[1], num_strings, size_string)
        train_strings = dataset.reshape(dataset.shape[0], dataset.shape[1], num_strings, size_string)
        sample_strings = sample_strings.permute(2, 0, 1, 3).reshape(num_strings, samples.shape[0], -1) # (num_strings, B, v*string_size)
        train_strings = train_strings.permute(2, 0, 1, 3).reshape(num_strings, dataset.shape[0], -1)   # (num_strings, P, v*string_size)
        frac_copies = (torch.einsum("aik, ajk -> aij", sample_strings, train_strings) == size_string).sum(dim=(2))
        frac_copies = (frac_copies > 0).int() # (num_strings, B). It takes into account that a given string can appear in many training points
        frac_copies = frac_copies.sum(-1) / samples.shape[0] # (num_strings,)
        hierarhical_copies[layer] = frac_copies.cpu()

    return hierarhical_copies


def compare_with_trainset(trainset, samples):
    """
    Check if the generated data is consistent with the rules.

    Args:
        trainset (torch.Tensor): dataset used for training. Shape (P, v, d)
        samples (torch.Tensor): generated samples. Shape (B, v, d)
        
    Returns:
        frac_copies (float): fraction of samples that are copies of the training set.
        mem_indices (torch.Tensor): indices of elements of the training set that have been generated.
    """
    # samples = torch.argmax(samples, dim=1).reshape(samples.shape[0], -1)
    # dataset = dataset.argmax(dim=1).reshape(dataset.shape[0], -1)
    # hamming = (samples != dataset).float().mean()

    d = trainset.shape[2]

    # print(samples.argmax(1)[:10])

    samples = samples.reshape(samples.shape[0], -1) # B, v*d
    trainset = trainset.reshape(trainset.shape[0], -1) # P, v*d

    matches = (samples @ trainset.T == d)
    frac_copies = matches.any(dim=1).float().mean()

    # training samples that are memorized by at least one generated sample
    mem_mask = matches.any(dim=0)
    mem_indices = torch.where(mem_mask)[0]

    return frac_copies.item(), mem_indices.cpu()


def compute_d3pm_loss_per_time(n_windows, points_per_window, model, x0, bp=False):

    with torch.no_grad():
        model.eval()

        time_losses = {}
        for time_window in range(n_windows):
            n_trajectories = points_per_window
            v, d = x0.shape[1], x0.shape[2]
            x = (
                x0.unsqueeze(1)
                .repeat(1, n_trajectories, 1, 1)
                .view(-1, v, d)
                .to(x0.device)
            )
            B = x.shape[0]
            _ts = torch.randint(
                model.n_T // n_windows * time_window,
                model.n_T // n_windows * (time_window + 1),
                (B,),
            ).to(x.device)

            proba = model.alphabar_t[_ts, None, None] * x + (
                1 - model.alphabar_t[_ts, None, None]
            ) / v * torch.ones_like(x)
            proba = proba.permute(0, 2, 1).reshape(-1, v)

            x_t = torch.multinomial(proba, num_samples=1)
            x_t = nn.functional.one_hot(x_t, v)
            x_t = x_t.reshape(-1, d, v).permute(0, 2, 1)

            if model.model_type == "start":
                if not bp:
                    true_p = model.proba_posterior_t_1(x_t, _ts, x)
                else:
                    nu_up_L = model.proba_posterior_0_t(x_t, _ts) # B, v, d
                    nu_up_L = nu_up_L.permute(1, 0, 2) # v, B, d
                    nu_up_L = nu_up_L.flatten(start_dim=1) # v, B*d
                    marginals = bp.run_BP_from_upward_messages(nu_up_L) # v, B*d
                    L = max(marginals.keys())
                    mean_x0_t = marginals[L] # v, B*d
                    mean_x0_t = mean_x0_t.reshape(v, B, d)
                    mean_x0_t = mean_x0_t.permute(1, 0, 2) # B, v, d
                    true_p = model.proba_posterior_t_1(x_t, _ts, mean_x0_t) # B, v, d
                true_p = true_p.permute(0, 2, 1).reshape(-1, true_p.shape[1]) # B*d, v
                true_p = true_p / true_p.sum(1, keepdim=True)
                model_p = model.proba_posterior_t_1(
                    x_t, _ts, model.readout(model.model(x_t, _ts / model.n_T))
                ) # B, v, d
                model_p = model_p.permute(0, 2, 1).reshape(-1, model_p.shape[1]) # B*d, v
                model_p = model_p / model_p.sum(1, keepdim=True)
                log_model_p = torch.log(model_p + 1e-8)
                log_true_p = torch.log(true_p + 1e-8)
                t_loss = (true_p * (log_true_p - log_model_p)).sum() / model_p.shape[0]
                time_losses[time_window] = t_loss.item()

            else:

                raise NotImplementedError

        return time_losses


def check_rules(samples, bp, trainset):
    """
    Check if the generated data is consistent with the rules.

    Args:
        samples (torch.Tensor): generated samples in one-hot encoding. Shape (B, v, d)
        bp (BP object): belief propagation object

    Returns:
        frac_correct (float): fraction of samples that are consistent with the rules.
        frac_valid_per_layer (dict): fraction of samples that are consistent with the rules for every layer.
    """

    x = torch.argmax(samples, dim=1) # B, d
    frac_correct, compatible_positions = BP_countcorrect_upward(x, bp)
    frac_valid = frac_correct[0].item()

    # reverse keys count the layers starting from the leaves (layer 0)
    frac_valid_per_layer = {bp.L-key : frac_correct[key].cpu() for key in frac_correct.keys()}
    frac_valid_per_layer = dict(sorted(frac_valid_per_layer.items()))

    compatible_positions = {bp.L-key : compatible_positions[key] for key in compatible_positions.keys()}
    compatible_positions = dict(sorted(compatible_positions.items()))

    total_valid_per_rule_layer, total_mem_per_rule_layer = process_sample_tuples(trainset, samples, compatible_positions,
                                                                                 bp.s, bp.m * bp.v)

    return frac_valid, frac_valid_per_layer, total_valid_per_rule_layer, total_mem_per_rule_layer



def compute_distance_histogram(samples):
    """
    Compute the histogram of distances between samples.

    Args:
        samples (torch.Tensor): generated samples in one-hot encoding. Shape (B, v, d)

    Returns:
        distances (torch.Tensor): histogram of distances between samples.
    """

    B = samples.size(0)
    d = samples.size(2)

    x1h = samples.reshape(B, -1) # B, v*d
    cdist = x1h @ x1h.T
    triu_indices = torch.triu_indices(B, B, offset=1)
    cdist = cdist[triu_indices[0], triu_indices[1]]
    cdist = cdist.cpu()

    bins = np.arange(0, d+2, 1)-0.5

    histo, _ = np.histogram(d - cdist, bins=bins)

    return histo


def compute_weight_norm_var(model, model0):

    with torch.no_grad():

        model.eval()

        deltaWeight_norms = {}
        weight_norms = {}
        for (name, p), (_, p0) in zip(model.named_parameters(), model0.named_parameters()):
            if p.requires_grad:
                deltaWeight_norms[name] = torch.norm(p - p0).item()
                weight_norms[name] = torch.norm(p).item()
        return {"dw": deltaWeight_norms, "w": weight_norms}


def get_tuples_in_dataset(dataset, test_layer, tuple_size):
    d = dataset.shape[2]
    v = dataset.shape[1]

    size_string = tuple_size ** test_layer
    num_strings = d // size_string

    trainset = dataset.reshape(dataset.shape[0], v, num_strings, size_string)
    trainset = trainset.permute(0, 2, 1, 3).reshape(-1, v, size_string)

    flat_trainset = trainset.reshape(trainset.shape[0], -1)
    unique_substrings, counts = torch.unique(flat_trainset, dim=0, return_counts=True)

    return unique_substrings.reshape(unique_substrings.shape[0], -1, size_string), counts


def get_rule_tuples(coords, samples, test_layer, tuple_size):
    """
    Creates a dict with key rule_idx and value all valid tuples at test_layer
    Returns:
    dict[rule_id] -> list of Torch tensors (tuples for each rule)
    """

    rule_dict = {}

    for rule_idx, batch_idx, pos_idx in coords:
        rule_idx, batch_idx, pos_idx = rule_idx.item(), batch_idx.item(), pos_idx.item()
        # get tuples from samples relevant to rule_idx
        datapoint = samples[batch_idx]
        tuples_datapoint = datapoint.reshape(datapoint.shape[0], -1, tuple_size ** test_layer)
        relevant = tuples_datapoint[:, pos_idx]

        # leave empty if no tuples
        if rule_idx not in rule_dict:
            rule_dict[rule_idx] = []

        rule_dict[rule_idx].append(relevant.unsqueeze(0))
    return rule_dict


def compare_rows(input_samples, reference_samples):
    eq = (input_samples[:, None] == reference_samples[None, :])
    row_match = eq.all(dim=(2, 3))
    mask = row_match.any(dim=1)
    return mask.int()


def get_rule_tuple_stats(rule_dict, trainset_tuples, num_rules):
    """
    Finds the total number of valid and memorized tuples per rule
    Returns:
    val: torch Tensor, size (num_rules,)
    mem: torch Tensor, size (num_rules,)
    """

    val = torch.zeros((num_rules,))
    mem = torch.zeros(num_rules)

    for rule, tensors in rule_dict.items():

        collected = torch.cat(tensors, dim=0)
        mask = compare_rows(collected, trainset_tuples)

        val[rule] = collected.shape[0]
        mem[rule] = mask.sum().item()

    return val, mem


def process_sample_tuples(trainset, samples, compatible_positions, tuple_size, num_rules):
    '''
    Computes total valid and memorized tuples per rule at all levels of the hierarchy
    Returns:
    valid_tuples: dict, {level_idx: torch.Tensor}
    memorized_tuples: dict, {level_idx: torch.Tensor}
    '''
    valid_tuples = {}
    memorized_tuples = {}

    for test_layer, coords in compatible_positions.items():
        trainset_tuples, _ = get_tuples_in_dataset(trainset, test_layer, tuple_size)

        # collect rule tuples
        rule_dict = get_rule_tuples(coords, samples, test_layer, tuple_size)

        # compute stats
        val, mem = get_rule_tuple_stats(rule_dict, trainset_tuples, num_rules)
        valid_tuples[test_layer] = val
        memorized_tuples[test_layer] = mem

    return valid_tuples, memorized_tuples



