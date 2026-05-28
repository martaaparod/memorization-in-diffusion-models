# RHM Experiments

---

## Running experiments

An example of the script used is available at [run.sh](diffusion_rhm/run.sh). The code (must be run from within the [diffusion_rhm](diffusion_rhm) directory) can be 
used to:
1. Train diffusion models and save checkpoints of the U-Net. At each evaluation step, sample a batch of generated data 
points and compute summary statistics, including:
   - Fraction of generated samples that are exact copies of the training set
   - Fraction of valid samples
   - Indices of training examples that appear in the generated batch
   - Number of tuples that satisfy each rule and layer in the RHM
   - Number of tuples that are memorized for each rule and layer in the RHM

Additionally, we record the rule-wise frequency of string tuples in the train set for each rule and layer in the RHM, 
as well as the log-probability of each training set element.

2. Sample data points from existing checkpoints in the results directory and store the statistics 
of the generated batch.

---

## Relevant Parameters

- `--zipf_exponent`: Zipf law exponent
- `--zipf_layer`: layer in the RHM at which the Zipf distribution is inserted (all other layers are uniformly 
distributed)
- `--unique`: whether enforcing uniqueness of samples when introducing Zipf's law in one of the layers of the hierarchy
- `--resume_checkpoint`: whether resuming training from the latest checkpoint that corresponds to a complete epoch