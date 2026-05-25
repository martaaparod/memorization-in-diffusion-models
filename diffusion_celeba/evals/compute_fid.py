import os
import shutil
import tempfile
from pathlib import Path
import torch
import torchvision.utils as vutils
from pytorch_fid import fid_score


def create_temp_image_dir(base_dir):
    '''
    Creates a temporary directory in which to save the sampled images as PNG
    '''
    temp_dir = tempfile.mkdtemp(prefix="fid_tmp_", dir=base_dir)
    return temp_dir


def save_samples_as_images(samples, temp_img_dir, device='cpu'):
    '''
    Save each image in samples to temp_img_dir as PNG
    Args:
    samples: torch tensor, containing sampled images
    tem_img_dir: str, path to directory in which to save PNG images
    '''
    counter = 0 
  
    for img in samples:
        filename = os.path.join(temp_img_dir, f"{counter:06d}.png")
        vutils.save_image(img, filename)
        counter += 1
    # return the total number of images saved
    return counter


def compute_fid(epoch, all_samples, base_dir, stats_file='evals/stats_celeba_64.npz', device='cuda'):
    '''
    Compute FID for a checkpoint
    stats_file: str, path to precomputed stats .npz for 50000 test CelebA images
    device: str, 'cuda' or 'cpu'
    '''
    base_dir = Path(base_dir)
    temp_img_dir = create_temp_image_dir(base_dir)

    # save images to temporary directory
    c = save_samples_as_images(all_samples, temp_img_dir)
    print(f'Saved {c} files in {temp_img_dir}')

    # compute fid
    fid = fid_score.calculate_fid_given_paths(
        [temp_img_dir, stats_file],
        batch_size=128,
        device=device,
        dims=2048
    )

    print(f"Epoch {epoch}: FID = {fid:.3f}")

    # delete temporary directory
    temp_img_dir = Path(temp_img_dir)
    if temp_img_dir.exists():
        shutil.rmtree(temp_img_dir)
    
    return fid
