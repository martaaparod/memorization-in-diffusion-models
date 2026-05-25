import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
from pathlib import Path
from dataset import CelebADataset
from diffusion_model import Diffusion
from train import train_diffusion, eval_checkpoint
from evals.memorization import build_training_tensor



def main(args):
    # set random seed
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create train and test dataset
    dataset = CelebADataset(args.data_dir)
    N = len(dataset) - 50000 # exclude last 50000 images (used for FID test set)
    indices = torch.randperm(N).tolist()

    if args.train_size is not None:
        train_size = args.train_size
        test_size = int(train_size * 0.2)
    
        train_idx = indices[:train_size]
        test_idx = indices[train_size:train_size + test_size]
    else:
        train_size = int((5 / 6) * N)
    
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # initialize model
    model = Diffusion(timesteps=args.T, model_size=args.model_size, device=device)

    if args.task == 'train':
        resume_epoch = None

        # for loading existing checkpoint
        if args.resume_training:
            root = Path(f'{args.results_dir}/train_size{args.train_size}')
            # find all epoch directories of the form epoch_*
            epoch_dirs = [d for d in root.glob('epoch_*') if d.is_dir()]

            if not epoch_dirs:
                raise ValueError('No epoch directories found!')

            # sort by epoch number in descending order and find latest epoch
            epoch_dirs.sort(key=lambda d: int(d.name.split('_')[1]), reverse=True)
            latest_epoch_dir = epoch_dirs[0]
            model_path = latest_epoch_dir / 'unet.pt'

            if not model_path.exists():
                raise FileNotFoundError(f'No checkpoint found at {model_path}')

            print(f'Loading latest checkpoint from {model_path}')
            model.unet.load_state_dict(torch.load(model_path, map_location=model.device))
            resume_epoch = int(latest_epoch_dir.name.split('_')[1])

        print('Setup complete. Ready to train.')
        train_diffusion(model, train_dataset, test_loader, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                        save_dir=f'{args.results_dir}/train_size{args.train_size}', resume_epoch=resume_epoch,
                       sample_batch_size=args.sample_batch_size, sample_n_batches=args.sample_n_batches)
        
    elif args.task == 'eval':
        X = build_training_tensor(train_dataset, device=model.device, batch_size=512)
        print('Begin loading checkpoints')

        root = Path(f'{args.results_dir}/train_size{args.train_size}')

        # iterate through all epoch_* folders
        for epoch_dir in sorted(root.glob('epoch_*')):
            epoch = int(epoch_dir.name.split('_')[1])
            model_path = epoch_dir / 'unet.pt'
            
            if not model_path.exists():
                continue

            print(f'Loading {model_path}')
            model.unet.load_state_dict(torch.load(model_path, map_location=model.device))

            eval_checkpoint(epoch, model, X, args.sample_batch_size, args.sample_n_batches, args.results_dir, save_unet=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model on CelebA with memorization metrics")

    # data and paths
    parser.add_argument('--data_dir', type=str, default='img_align_celeba', help='path to CelebA folder')
    parser.add_argument('--results_dir', type=str, default='results', help='directory to save results')

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--epochs', type=int, default=150000, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--task', type=str, default='train', help='if training or sampling from checkpoints')
    parser.add_argument('--train_size', type=int, default=None, help='train size')
    parser.add_argument('--resume_training', action='store_true', help='if resuming training from latest checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--model_size', type=str, default='large', help='model size for U-net')
    parser.add_argument('--T', type=int, default=1000, help='number of timesteps for training DPPM')

    # sampling hyperparameters
    parser.add_argument('--sample_batch_size', type=int, default=2000, help='number of images sampled in one go')
    parser.add_argument('--sample_n_batches', type=int, default=25, help='number of sampling batches per checkpoint')

    args = parser.parse_args()
    main(args)
