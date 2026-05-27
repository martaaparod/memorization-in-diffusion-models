# Diffusion Models Preferentially Memorize Prototypical Examples or: Why Does My Diffusion Model Love Slop?

This repository contains the code to run the experiments from the paper. 
The repository contains code to train a diffusion model and run experiments in 
the RHM and CelebA.

---
## Random Hierarchy Model (RHM)
Derived from the repository [https://github.com/AntonioScl/minimal_diffusion_rhm](https://github.com/AntonioScl/minimal_diffusion_rhm).
Contains:
- Code for building the RHM variants with Zipf distribution and uniqueness of samples
- Code for training Discrete Diffusion models
- Code for evaluations of memorized and valid samples and subtuples
- Code to compute the log-likelihood of training and generated data

---
## CelebA

Contains:
- Code for training models
- Architectures used for the U-Net
- Code for evaluations of generalization (FID), memorization and log-likelihood estimation of sampled images
