from itertools import *
import warnings
import copy
import sys

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import dec2bin, dec2base, base2dec


def sample_rules( v, n, m, s, L, seed=42):
        """
        Sample random rules for a random hierarchy model.

        Args:
            v: The number of values each variable can take (vocabulary size, int).
            n: The number of classes (int).
            m: The number of synonymic lower-level representations (multiplicity, int).
            s: The size of lower-level representations (int).
            L: The number of levels in the hierarchy (int).
            seed: Seed for generating the rules.

        Returns:
            A dictionary containing the rules for each level of the hierarchy.
        """
        random.seed(seed)
        tuples = list(product(*[range(v) for _ in range(s)]))

        rules = {}
        rules[0] = torch.tensor(
                random.sample( tuples, n*m)
        ).reshape(n,m,-1)
        for i in range(1, L):
            rules[i] = torch.tensor(
                    random.sample( tuples, v*m)
            ).reshape(v,m,-1)

        return rules


def sample_data_from_labels(labels, rules, probability):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen according to probability.

    Args:
        labels: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        probability: A dictionary containing the distribution of the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs, outputs, rule frequencies and log-likelihood of data points.
    """
    L = len(rules)  # number of levels in the hierarchy
    features = labels

    # infer v, n, m and initialize frequency dict
    n = rules[0].shape[0]
    m = rules[0].shape[1]
    v = rules[1].shape[0] if L > 1 else rules[0].shape[0]
    rule_freqs = {l: torch.zeros(((n if l == 0 else v) * m), dtype=torch.long) for l in range(L)}

    total_logprob_per_sample = torch.zeros(features.shape[0], dtype=torch.float)

    for l in range(L):
        chosen_rule = torch.multinomial(probability[l], features.numel(), replacement=True).reshape(features.shape)
        chosen_probs = probability[l][chosen_rule]

        if chosen_probs.ndim == 1:
            layer_logprob = chosen_probs.log()
        else:
            layer_logprob = chosen_probs.log().sum(dim=1)

        total_logprob_per_sample += layer_logprob

        # record rule frequencies (flattened from 0 to v*m)
        i = features.long().reshape(-1)
        j = chosen_rule.long().reshape(-1)
        vm = (n if l == 0 else v) * m
        rule_freqs[l] += torch.bincount(i * m + j, minlength=vm)

        features = rules[l][features, chosen_rule].flatten(start_dim=1)

    # reverse indices
    rule_freqs = {L - key: rule_freqs[key] for key in rule_freqs.keys()}

    return features, labels, rule_freqs, total_logprob_per_sample


def generate_labels(size, num_classes):
    '''
    Uniformly samples labels
    '''
    return torch.randint(low=0, high=num_classes, size=(size,))


def sample_data_from_labels_per_sample(labels, rules, probability):
    """
    Samples a batch of data points, returning per sample statistics
    Returns:
    features: [batch_size, ...]
    labels: [batch_size]
    rule_freqs_per_sample: list of length L, each entry is [batch_size, vm]
    logprob: [batch_size]
    """

    L = len(rules)
    features = labels
    batch_size = features.shape[0]
    n = rules[0].shape[0]
    m = rules[0].shape[1]
    v = rules[1].shape[0] if L > 1 else rules[0].shape[0]

    # per-sample logprob
    logprob = torch.zeros(batch_size)

    # for storing rule frequencies per sample
    rule_freqs_per_sample = []

    for l in range(L):
        chosen_rule = torch.multinomial(probability[l], features.numel(), replacement=True).reshape(features.shape)
        chosen_probs = probability[l][chosen_rule]

        if chosen_probs.ndim == 1:
            layer_logprob = chosen_probs.log()
        else:
            layer_logprob = chosen_probs.log().sum(dim=1)
        logprob += layer_logprob

        # add rule frequency per sample
        vm = (n if l == 0 else v) * m
        i = features.reshape(batch_size, -1)
        j = chosen_rule.reshape(batch_size, -1)
        idx = i * m + j

        freq = torch.zeros(batch_size, vm, dtype=torch.long)
        freq.scatter_add_(1, idx, torch.ones_like(idx))
        rule_freqs_per_sample.append(freq)

        # hierarchy transition
        features = rules[l][features, chosen_rule].flatten(start_dim=1)

    return features, labels, rule_freqs_per_sample, logprob


def filter_unique(features, labels, logprob, rule_freqs_per_sample, seen):
    """
    Removes duplicate feature rows and filters all associated statistics
    Keeps first occurrence of each unique feature
    """

    batch_size = features.shape[0]

    # flatten features for uniqueness check
    flat = features.view(batch_size, -1).detach().to('cpu')

    keep_indices = []

    for i in range(batch_size):
        key = hash(flat[i].numpy().tobytes())

        if key not in seen:
            seen.add(key)
            keep_indices.append(i)

    if len(keep_indices) == 0:
        return None

    keep_indices = torch.tensor(keep_indices, dtype=torch.long)

    # filter statistics
    features = features[keep_indices]
    labels = labels[keep_indices]
    logprob = logprob[keep_indices]

    rule_freqs_per_sample = [
        r[keep_indices] for r in rule_freqs_per_sample
    ]

    return features, labels, rule_freqs_per_sample, logprob


def sample_unique_data_from_labels(target_size, num_classes, rules, probability):
    """
    Generates target_size unique samples from RHM according to the sampling rules and probability
    1. Generates a batch
    2. Keeps unique elements
    2. Generates a batch of missing size
    3. Repeats steps until a unique batc is created
    """
    seen = set()

    all_features = []
    all_labels = []
    all_logprob = []
    all_rule_freqs = []

    collected = 0

    while collected < target_size:

        remaining = target_size - collected

        # generate labels
        labels = generate_labels(remaining, num_classes)

        # sample batch
        features, labels, rule_freqs, logprob = sample_data_from_labels_per_sample(labels, rules, probability)

        # filter unique
        out = filter_unique(features, labels, logprob, rule_freqs, seen)

        if out is None:
            continue

        features, labels, rule_freqs, logprob = out

        # store
        all_features.append(features)
        all_labels.append(labels)
        all_logprob.append(logprob)
        all_rule_freqs.append(rule_freqs)

        collected += features.shape[0]

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_logprob = torch.cat(all_logprob, dim=0)

    # sum rule frequencies for data points that are not rejected
    final_rule_freqs = {}
    L = len(all_rule_freqs[0])

    for l in range(L):
        freq_l = torch.cat([x[l] for x in all_rule_freqs], dim=0).sum(dim=0)
        final_rule_freqs[l] = freq_l

    return all_features, all_labels, final_rule_freqs, all_logprob


def sample_data_from_labels_unif( labels, rules, bonus):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen uniformly at random for each level.

    Args:
        labels: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model, as well as rule frequencies.
    """
    L = len(rules)  # number of levels in the hierarchy
    features = labels

    # infer v, n, m and initialize frequency dict
    n = rules[0].shape[0]
    m = rules[0].shape[1]
    v = rules[1].shape[0] if L > 1 else rules[0].shape[0]
    rule_freqs = {l: torch.zeros(((n if l == 0 else v) * m), dtype=torch.long) for l in range(L)}

    if bonus:		# extra output for additional measures
        if 'size' not in bonus.keys():
            bonus['size'] = samples.size(0)
        if 'tree' in bonus:
            tree = {}
            bonus['tree'] = tree
        if 'noise' in bonus:	# add corrupted versions of the last bonus[-1] data
            noise = {}
            noise[L] = copy.deepcopy(features[-bonus['size']:])	# copy current representation (labels)...
            noise[L][:] = torch.randint(rules[0].shape[0], (bonus['size'],))	# ...and randomly change it
            bonus['noise'] = noise
        if 'synonyms' in bonus:	# add synonymic versions of the last bonus[-1] data
            synonyms = {}
            bonus['synonyms'] = synonyms

    for l in range(L):
        # choose random rule for each element in the level
        chosen_rule = torch.randint(low=0, high=rules[l].shape[1], size=features.shape)

        # record rule frequencies (flattened from 0 to v*m)
        i = features.long().reshape(-1) 
        j = chosen_rule.long().reshape(-1) 
        vm = (n if l == 0 else v) * m
        rule_freqs[l] += torch.bincount(i * m + j, minlength=vm)

        if bonus:
            if 'tree' in bonus:
                tree[L-l] = copy.deepcopy(features[-bonus['size']:])

            if 'synonyms' in bonus:

                for ell in synonyms.keys():	# propagate modified data down the tree TODO: randomise whole downstream propagation
                    synonyms[ell] = rules[l][synonyms[ell], chosen_rule[-bonus['size']:]].flatten(start_dim=1)

                synon_rule =  copy.deepcopy(chosen_rule[-bonus['size']:]) 		# copy current representation indices...
                if l==0:
                    synon_rule[:] = torch.randint(rules[l].shape[1], (synon_rule.size(0),))		# ... and randomly change it (only one index at the highest level)
                else:
                    synon_rule[:,-2] = torch.randint(rules[l].shape[1], (synon_rule.size(0),))	# ... and randomly change the next-to-last

                synonyms[L-l] =  copy.deepcopy(features[-bonus['size']:])
                synonyms[L-l] = rules[l][synonyms[L-l], synon_rule].flatten(start_dim=1)
                #TODO: add custom positions for 'synonyms'

        features = rules[l][features, chosen_rule].flatten(start_dim=1)                 # Apply the chosen rule to each variable in the current level

        if bonus:
            if 'noise' in bonus:

                for ell in noise.keys():	# propagate modified data down the tree TODO: randomise whole downstream propagation
                    noise[ell] = rules[l][noise[ell], chosen_rule[-bonus['size']:]]
                    noise[ell] = noise[ell].flatten(start_dim=1)

                noise[L-l-1] =  copy.deepcopy(features[-bonus['size']:])	# copy current representation ...
                noise[L-l-1][:,-2] = torch.randint(rules[l].shape[0], (bonus['size'],))     # ... and randomly change the next-to-last feature
                #TODO: rules[l].shape[0] not v in general!!! FIX IT!
                #TODO: add custom positions for 'noise'

    # reverse indices
    rule_freqs = {L-key : rule_freqs[key] for key in rule_freqs.keys()}

    return features, labels, rule_freqs


def sample_data_from_indices(samples, rules, v, n, m, s, L, bonus):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).
        bonus: Dictionary for additional output (list), includes 'noise' (randomly replace one symbol at each level), 'synonyms' (randomply resample one production rule at each level), 'tree' (stores the data derivation), 'size' (number of bonus data).

    Returns:
        A tuple containing the inputs and outputs of the model (plus additional output in bonus dict).
    """
    max_data = n * m ** ((s**L-1)//(s-1))
    data_per_hl = max_data // n 	# div by num_classes to get number of data per class

    high_level = samples.div(data_per_hl, rounding_mode='floor')	# div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl					# compute remainder (run in range(data_per_hl))

    labels = high_level		# labels are the classes (features of highest level)
    features = labels		# init input features as labels (rep. size 1)
    size = 1

    if bonus:		# extra output for additional measures
        if 'size' not in bonus.keys():
            bonus['size'] = samples.size(0)
        if 'tree' in bonus:
            tree = {}
            bonus['tree'] = tree
        if 'noise' in bonus:	# add corrupted versions of the last bonus[-1] data
            noise = {}
            noise[L] = copy.deepcopy(features[-bonus['size']:])	# copy current representation (labels)...
            noise[L][:] = torch.randint(n, (bonus['size'],))	# ...and randomly change it
            bonus['noise'] = noise
        if 'synonyms' in bonus:	# add synonymic versions of the last bonus[-1] data
            synonyms = {}
            bonus['synonyms'] = synonyms

    for l in range(L):

        choices = m**(size)
        data_per_hl = data_per_hl // choices	# div by num_choices to get number of data per high-level feature

        high_level = low_level.div( data_per_hl, rounding_mode='floor') # div by data_per_hl to get high-level feature index (1 index in range(m**size))
        high_level = dec2base(high_level, m, length=size).squeeze()     # convert to base m (size indices in range(m), squeeze needed if index already in base m)

        if bonus:
            if 'tree' in bonus:
                tree[L-l] = copy.deepcopy(features[-bonus['size']:])

            if 'synonyms' in bonus:

                for ell in synonyms.keys():	# propagate modified data down the tree TODO: randomise whole downstream propagation
                    synonyms[ell] = rules[l][synonyms[ell], high_level[-bonus['size']:]]
                    synonyms[ell] = synonyms[ell].flatten(start_dim=1)

                high_level_syn =  copy.deepcopy(high_level[-bonus['size']:]) 			# copy current representation indices...
                if l==0:
                    high_level_syn[:] = torch.randint(m, (high_level_syn.size(0),))		# ... and randomly change it (only one index at the highest level)
                else:
                    high_level_syn[:,-2] = torch.randint(m, (high_level_syn.size(0),))	# ... and randomly change the next-to-last
                synonyms[L-l] =  copy.deepcopy(features[-bonus['size']:])
                synonyms[L-l] = rules[l][synonyms[L-l], high_level_syn]
                synonyms[L-l] = synonyms[L-l].flatten(start_dim=1)
                #TODO: add custom positions for 'synonyms'
        
        features = rules[l][features, high_level]	        		# apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
        features = features.flatten(start_dim=1)				# flatten to tensor of size (batch_size, size*s)
        size *= s								# rep. size increases by s at each level
        low_level = low_level % data_per_hl					# compute remainder (run in range(data_per_hl))

        if bonus:
            if 'noise' in bonus:

                for ell in noise.keys():	# propagate modified data down the tree TODO: randomise whole downstream propagation
                    noise[ell] = rules[l][noise[ell], high_level[-bonus['size']:]]
                    noise[ell] = noise[ell].flatten(start_dim=1)

                noise[L-l-1] =  copy.deepcopy(features[-bonus['size']:])	# copy current representation ...
                noise[L-l-1][:,-2] = torch.randint(v, (bonus['size'],))	# ... and randomly change the next-to-last feature
                #TODO: add custom positions for 'noise'

    return features, labels


class RandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
            self,
            num_features=8,     # vocavulary size
            num_classes=2,      # number of classes
            num_synonyms=2,     # number of synonymic low-level representations (multiplicity)
            tuple_size=2,       # size of the low-level representations
            num_layers=2,       # number of levels in the hierarchy
            probability=None,   # for assigning nonuniform probabilities to production rules
            seed_rules=0,
            seed_sample=1,
            train_size=-1,
            test_size=0,
            input_format='onehot',
            whitening=0,
            transform=None,
            replacement=False,
            bonus={},
            unique=True
    ):

        self.num_features = num_features
        self.num_synonyms = num_synonyms 
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tuple_size = tuple_size
        self.rule_freqs = None
        self.total_logprob_per_sample = None

        self.rules = sample_rules(num_features, num_classes, num_synonyms, tuple_size, num_layers, seed=seed_rules)
 
        max_data = num_classes * num_synonyms ** ((tuple_size ** num_layers - 1) // (tuple_size - 1))
        assert train_size >= -1, "train_size must be greater than or equal to -1"

        if max_data > sys.maxsize and not replacement:
            print(
                "Max dataset size cannot be represented with int64! Using sampling with replacement."
            )
            warnings.warn(
                "Max dataset size cannot be represented with int64! Using sampling with replacement.",
                RuntimeWarning,
            )
            replacement = True

        if not replacement:

            assert probability is None, "nonuniform probability only implemented for sampling with replacement."
            if train_size == -1:
                samples = torch.arange( max_data)

            else:
                test_size = min( test_size, max_data-train_size)
                random.seed(seed_sample)
                samples = torch.tensor( random.sample( range(max_data), train_size+test_size))

            self.features, self.labels = sample_data_from_indices(
                samples, self.rules, num_features, num_classes, num_synonyms, tuple_size, num_layers, bonus
            )

        else:

            # TODO: implement synonymic and noisy data for sampling with replacement
            torch.manual_seed(seed_sample)
            if train_size == -1:
                labels = torch.randint(low=0, high=num_classes, size=(max_data + test_size,))
            else:
                labels = torch.randint(low=0, high=num_classes, size=(train_size + test_size,))
            if probability is None:
                self.features, self.labels, self.rule_freqs = sample_data_from_labels_unif(
                    labels, self.rules, bonus
                )
            else:   # TODO: implement synonymic and noisy data for arbitrary distribution
                self.probability = probability
                if unique:
                    self.features, self.labels, self.rule_freqs, self.total_logprob_per_sample = (
                        sample_unique_data_from_labels(
                            train_size + test_size,
                            num_classes,
                            self.rules,
                            self.probability,
                        )
                    )
                else:
                    self.features, self.labels, self.rule_freqs, self.total_logprob_per_sample = sample_data_from_labels(
                        labels, self.rules, self.probability
                    )


        if 'onehot' not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

        if 'tuples' in input_format:
            self.features = base2dec(self.features.view(self.features.size(0), -1, tuple_size), num_features)
            if bonus:
                if 'synonyms' in bonus:
                    for k in bonus['synonyms'].keys():
                        bonus['synonyms'][k] = base2dec(bonus['synonyms'][k].view(bonus['synonyms'][k].size(0), -1, tuple_size), num_features)

                if 'noise' in bonus:
                    for k in bonus['noise'].keys():
                        bonus['noise'][k] = base2dec(bonus['noise'][k].view(bonus['synonyms'][k].size(0), -1, tuple_size), num_features)

        if 'onehot' in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=num_features if 'tuples' not in input_format else num_features**tuple_size
            ).float()
            if bonus:
                if 'synonyms' in bonus:
                    for k in bonus['synonyms'].keys():
                        bonus['synonyms'][k] = F.one_hot(
                            bonus['synonyms'][k].long(),
                            num_classes=num_features if 'tuples' not in input_format else num_features**tuple_size
                        ).float()
                        bonus['synonyms'][k] = bonus['synonyms'][k].permute(0, 2, 1)
                if 'noise' in bonus:
                    for k in bonus['noise'].keys():
                        bonus['noise'][k] = F.one_hot(
                            bonus['noise'][k].long(),
                            num_classes=num_features if 'tuples' not in input_format else num_features**tuple_size
                        ).float()
                        bonus['noise'][k] = bonus['noise'][k].permute(0, 2, 1)

            if whitening:

                inv_sqrt_norm = (1.-1./num_features) ** -.5
                self.features = (self.features - 1./num_features) * inv_sqrt_norm
                if bonus:
                    if 'synonyms' in bonus:
                        for k in bonus['synonyms'].keys():
                            bonus['synonyms'][k] = (bonus['synonyms'][k] - 1./num_features) * inv_sqrt_norm

                    if 'noise' in bonus:
                        for k in bonus['noise'].keys():
                            bonus['noise'][k] = (bonus['noise'][k] - 1./num_features) * inv_sqrt_norm

            self.features = self.features.permute(0, 2, 1)

        elif 'long' in input_format:
            self.features = self.features.long() + 1

            if bonus:
                if 'synonyms' in bonus:
                    for k in bonus['synonyms'].keys():
                        bonus['synonyms'][k] = bonus['synonyms'][k].long() + 1

                if 'noise' in bonus:
                    for k in bonus['noise'].keys():
                        bonus['noise'][k] = bonus['noise'][k].long() + 1

        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
        	idx: sample index

        Returns:
            Feature-label pairs at index            
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

    def get_rules(self):
        return self.rules



