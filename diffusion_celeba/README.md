# CelebA experiments

---
## Setup

To train a diffusion model, you will need to download the CelebA dataset 
from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Ensure there is a folder 
containing all .jpg files in [diffusion_celeba](diffusion_celeba). The path to this directory
can be specified using `--data_dir`.

To evaluate the log-likelihood of generated samples, we use a fine-tuned ConvNeXt V2 nano model.
The checkpoint used is available here, and should be placed in [evals](diffusion_Celeba/evals).

---

## Running experiments

The code (must be run from within the [diffusion_celeba](diffusion_celeba) directory) can be 
used to:
1. Train diffusion models, save checkpoints of the U-Net, sample images and store the statistics of 
the generated batch:
    - FID score
    - Fraction of memorized images
    - Train indices of each memorized image
    - Log-likelihood of each generated image

```python
python main.py --task train --epochs 100000 --batch_size 512 --lr 1e-5 --train_size 10000
```
2. Sample images from existing checkpoints in the results directory and store the statistics 
of the generated batch.

```python
python main.py --task eval --train_size 10000
```

The number of sampled images per checkpoint can be adjusted using `--sample_batch_size` 
and `--sample_n_batches`, where the total number of images sampled will be `sample_batch_size`*`sample_n_batches`.

