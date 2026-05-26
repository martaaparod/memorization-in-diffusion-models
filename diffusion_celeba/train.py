import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from evals.compute_fid import compute_fid
from evals.memorization import compute_mem, build_training_tensor
from evals.classifier import compute_loglikelihood_samples

def is_power_of_n(x, n):
    '''
    Check if x is a power of n
    '''
    if x < 1:
        return False
    while x % n == 0:
        x //= n
    return x == 1


def eval_checkpoint(epoch, diffusion, X, sample_size, n_batches, save_dir, save_unet=True):
    '''
    Samples images and computes FID, fraction of memorized images, train indices memorized and log-likelihood of memorized images 
    and saves results
    Args:
    epoch: int, checkpoint epoch
    save_unet: bool, when True the unet state dict is saved
    '''
    # create directory for checkpoint
    epoch_dir = Path(save_dir) / f'epoch_{epoch}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    if save_unet:
        # save state dict of unet
        torch.save(diffusion.unet.state_dict(), f"{epoch_dir}/unet.pt")

    # sampling
    diffusion.unet.eval()
    all_samples = []
    
    print(f'Sampling at epoch {epoch}')
    for i in range(n_batches):
        with torch.no_grad():
            all_samples.append(diffusion.sample(sample_size).detach().cpu())
    all_samples = torch.cat(all_samples, dim=0)

    # analyze samples
    # fid
    fid = compute_fid(epoch, all_samples, base_dir=epoch_dir)
    # memorization
    frac_rel, train_indices = compute_mem(X, all_samples, n_batches, sample_size)
    # log-likelihood
    pll = compute_loglikelihood_samples(all_samples, batch_size=128)

    results_epoch = {'fid': fid,
                    'frac_rel': frac_rel,
                    'train_indices': train_indices,
                    'pll': pll
                    }
    
    # save stats
    logs_dir = Path(f'{save_dir}/logs.pt')
    if logs_dir.exists():
        res_dict = torch.load(logs_dir, map_location='cpu', weights_only=False)
    else:
        res_dict = {}

    res_dict[epoch] = results_epoch
    torch.save(res_dict, logs_dir)


def train_diffusion(diffusion, train_dataset, testloader, lr, epochs, sample_batch_size=2000,  sample_n_batches=25,
                    save_frequency=2, save_dir='results', batch_size=64, resume_epoch=None):
    # build optimizer
    optimizer = torch.optim.Adam(diffusion.unet.parameters(), lr=lr)

    X = build_training_tensor(train_dataset, device='cuda', batch_size=512)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_losses = []
    test_losses = []


    if resume_epoch is None:
        start_epoch = 0
    else:
        start_epoch = resume_epoch + 1

    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch}')
        diffusion.unet.train()
        train_loss_epoch = 0.0
        n_train_batches = 0

        for batch in tqdm(dataloader):
            # normalize imgs
            batch = batch.to(diffusion.device) * 2 - 1

            t = torch.randint(0, diffusion.T, (batch.shape[0],)).long().to(diffusion.device)
            
            # get train loss
            loss = diffusion.get_loss(batch, t)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            n_train_batches += 1

        train_loss_epoch /= n_train_batches
        train_losses.append(train_loss_epoch)

        print(f'Train loss: {train_loss_epoch:.6f}')

        # evaluate test loss
        diffusion.unet.eval()
        test_loss_epoch = 0.0
        n_test_batches = 0

        with torch.no_grad():
            for batch in testloader:
                # noramlize imgs
                batch = batch.to(diffusion.device) * 2 - 1

                t = torch.randint(0, diffusion.T, (batch.shape[0],)).long().to(diffusion.device)

                # get test loss
                loss = diffusion.get_loss(batch, t)

                test_loss_epoch += loss.item()
                n_test_batches += 1

            test_loss_epoch /= n_test_batches
            test_losses.append(test_loss_epoch)

            print(f'Test loss: {test_loss_epoch:.6f}')

        # take checkpoints
        if is_power_of_n(epoch, save_frequency) or (epoch == epochs - 1) or (epoch % 5000 == 0):
            eval_checkpoint(epoch, diffusion, X, sample_batch_size, sample_n_batches, save_dir)

    return train_losses, test_losses


