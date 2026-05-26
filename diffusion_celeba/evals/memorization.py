import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def build_training_tensor(dataset, device='cpu', batch_size=256):
    '''
    Builds flattened tensor containing train images from dataset object
    '''
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_imgs = []

    for batch in tqdm(loader, desc="Building training tensor"):
        all_imgs.append(batch)

    X = torch.cat(all_imgs, dim=0) # (N, C, H, W)
    X = X.reshape(X.shape[0], -1).float() # flatten
    return X.to(device)


def compute_knn(samples, X, k=2, chunk_size=1000):
    '''
    Computes k-smallest distances and k-nearest neighbours of elements in samples to elements in X
    Args:
    samples: torch tensor, flattened (B, C*H*W)
    X: torch tensor, flattened (N, C*H*W)
    chunk_size: int, compare distances to smaller batches of elements in X
    '''
    all_dists = []

    for i in range(0, X.shape[0], chunk_size):
        X_chunk = X[i:i+chunk_size]

        # cdist (B, chunk)
        dist_chunk = torch.cdist(samples, X_chunk)
        all_dists.append(dist_chunk)
    # combine (B, N)
    dist = torch.cat(all_dists, dim=1)
    
    knn_dists, knn_idx = dist.topk(k, dim=1, largest=False)
    return knn_dists, knn_idx


def compute_mem(X, all_samples, n_batches, sample_size, gap_threshold=1/3):
    '''
    Computes fraction of generated samples that memorize the training set
    Args:
    checkpoint_dir: str, directory to store results
    X: torch tensor, containing train images (flattened across non-batch dimension)
    all_samples: torch tensor, containing generated samples
    n_batches: int, to split all_samples into smaller subsets
    sample_size: int, to split all_samples into smaller subsets
    gap_threshold: float, threshold  in (0, 1) used in memorization definition
    '''
    k = 2

    distances_all = torch.zeros(n_batches * sample_size, k, device='cpu')
    knn_idx_all = torch.zeros(n_batches * sample_size, k, device='cpu')

    for i in range(n_batches):
        # split into batches for reduced memory usage
        start, end = i * sample_size, (i+1) * sample_size
        samples = all_samples[start:end]

        # flatten images
        samples_flat = samples.reshape(samples.shape[0], -1)

        # compute k-NN
        knn_dists, knn_idx = compute_knn(samples_flat, X.to(all_samples.device), k)

        distances_all[start:end, :] = knn_dists.cpu()
        knn_idx_all[start:end, :] = knn_idx.cpu()

    # apply relative memorization definition
    gap_ratio = distances_all[:, 0] / distances_all[:, 1]
    samples_rel = (gap_ratio < gap_threshold).nonzero(as_tuple=False)
    train_rel = knn_idx_all[samples_rel, 0].long() # indices of train images that are memorized
    frac_rel = len(samples_rel) / len(gap_ratio) # get fraction of memorized samples

    print(f'Memorization: {frac_rel}')
    return frac_rel, train_rel
