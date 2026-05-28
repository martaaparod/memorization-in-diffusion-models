from typing import Dict, Optional, Tuple
import torch
import pickle
import os
import subprocess
import init
from train import train
from pathlib import Path


def parse_args() -> Dict:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train DDPM on the Random Hierarchy Model"
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--grid", type=bool, default=False, help="use grid for training")
    parser.add_argument(
        "--eval", default=False, action="store_true", help="use for evaluating checkpoints")

    ### Process
    parser.add_argument(
        "--process",
        type=str,
        default="discrete",
        help="Process to use (implemented: continuous, discrete)",
    )
    parser.add_argument(
        "--nT",
        type=int,
        default=200,
        help="Number of time steps for the diffusion process",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.001, help="beta1 for the diffusion process"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.1, help="beta2 for the diffusion process"
    )

    ### Dataset parameters
    parser.add_argument(
        "--dataset", type=str, default="rhm", help="Dataset to reproduce"
    )
    parser.add_argument(
        "--num_features", metavar="v", type=int, help="number of features"
    )
    parser.add_argument(
        "--num_classes", metavar="n", type=int, help="number of classes", default=None
    )
    parser.add_argument(
        "--num_synonyms",
        metavar="m",
        type=int,
        help="multiplicity of low-level representations",
    )
    parser.add_argument(
        "--tuple_size", metavar="s", type=int, help="size of low-level representations"
    )
    parser.add_argument("--num_layers", metavar="L", type=int, help="number of layers")
    parser.add_argument("--seed_rules", type=int, help="seed for the dataset")
    parser.add_argument(
        "--train_size", metavar="Ptr", type=int, help="training set size"
    )
    parser.add_argument("--batch_size", metavar="B", type=int, help="batch size")
    parser.add_argument(
        "--test_size", metavar="Pte", default=256, type=int, help="test set size"
    )
    parser.add_argument(
        "--generate_all",
        default=False,
        action="store_true",
        help="generate all the dataset",
    )
    parser.add_argument(
        "--seed_sample", type=int, help="seed for the sampling of train and testset"
    )
    parser.add_argument(
        "--replacement",
        default=False,
        action="store_true",
        help="sample with replacement for the rhm dataset",
    )
    parser.add_argument("--input_format", type=str, default="onehot")
    parser.add_argument("--whitening", type=int, default=0)
    parser.add_argument("--zipf_exponent", type=int, help='Zipf law exponent', default=None)
    parser.add_argument("--zipf_layer", type=int, help='layer where Zipf law is introduced', default=None)
    parser.add_argument("--unique", default=False, action="store_true",
                        help="whether enforcing the data generation to be unique under Zipf distribution")

    """
    Architecture args
    """
    parser.add_argument(
        "--model", type=str, help="architecture (implemented: hUnet, bpUnet)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="start",
        help="If architecture predicts noise or the starting point",
    )
    parser.add_argument("--model_output", type=str, default="logits")
    parser.add_argument("--depth", type=int, default=None, help="depth of the network")
    parser.add_argument("--width", type=int, default=256, help="width of the network")
    parser.add_argument("--filter_size", type=int, default=None)
    parser.add_argument(
        "--num_heads", type=int, help="number of heads (transformer Unet only)"
    )
    parser.add_argument(
        "--embedding_dim", type=int, help="embedding dimension (transformer only)"
    )
    parser.add_argument("--bias", default=False, action="store_true")
    parser.add_argument("--seed_model", type=int, help="seed for model initialization")

    """
        Training args
    """
    parser.add_argument("--optim", type=str, default="adam", help="optimizer to use [adam, sgd]")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--accumulation", default=False, action="store_true")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--scheduler_time", type=int, default=None)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=1,
        help="number of forward process trajectories used in the loss",
    )
    parser.add_argument("--resume_checkpoint", default=False, action="store_true",
                        help="whether resuming training from latest epoch")

    """
	Output args
    """
    parser.add_argument("--print_period", type=int, help="period of prints, linearly spaced.", default=1000)
    parser.add_argument("--save_freq", type=int, help="frequency of saves, logarithmically spaced.", default=1)
    parser.add_argument("--loss_threshold", type=float, default=1e-3)
    parser.add_argument(
        "--output", type=str, required=True, help="path of the output file"
    )

    args = parser.parse_args()
    if args.num_classes is None:
        args.num_classes = args.num_features
    if args.depth is None:
        args.depth = args.num_layers
    if args.filter_size is None:
        args.filter_size = args.tuple_size
    if args.generate_all:
        Pmax = args.num_classes * args.num_synonyms ** (
            (args.tuple_size**args.num_layers - 1) // (args.tuple_size - 1)
        )
        if Pmax > 1e8:
            print("Pmax is too large, impossible to generate all the dataset")
            args.generate_all = False
        else:
            args.test_size = Pmax - args.train_size

    return args


def run_with_grid(train_iter):
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    with open(args.output, 'wb') as handle:
        pickle.dump(args, handle)
    saved = False
    try:
        for data in train_iter:
            data['git'] = git
            data['args'] = args
            with open(args.output, 'wb') as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
            saved = True
    except KeyboardInterrupt:
        if not saved:
            os.remove(args.output)
        raise


def sample_and_analyze_substrings(ddpm, args, trainloader, testloader, eval_func):
    log_step = {}

    steps_per_epoch = len(trainloader)

    ddpm.eval()
    with torch.no_grad():
        x = next(iter(trainloader))[0].to(args.device)
        x_test = next(iter(testloader))[0].to(args.device)
        xh = ddpm.sample(10000, (args.num_features, args.tuple_size**args.num_layers), args.device)

        for key, value in eval_func.items():
            if key == "Time_losses":
                val = value(ddpm, x)
            elif key in ["Test_losses", "True_losses"]:
                val = value(ddpm, x_test)
            elif key == "Weight_norm":
                val = value(ddpm.model, ddpm.model)
            else:
                val = value(xh)
                print(f"{key} : {val}", flush=True)
            log_step[int(key)/steps_per_epoch] = val

    return log_step


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    args = parse_args()

    # setup
    print('Initializing dataset')
    train_loader, test_loader, rules, dataset, rule_freqs, total_logprob_per_sample = init.init_data(args)
    ddpm = init.init_model(args)
    ddpm.to(args.device)
    optim_sched = init.init_optimizer(ddpm, args)
    bp = init.init_bp(args, rules)
    eval_func = init.init_eval_func(rules, bp, args, dataset)

    directory = Path(f'./results/{args.output}_ddpm_{args.dataset}_zipf{args.zipf_exponent}_layer{args.zipf_layer}')
    if not directory.exists():
        print(f"Directory {directory} does not exist.")

    # sample from existing checkpoints and evaluate generated strings
    if args.eval:
        dict_progress = torch.load(f'{directory}/logs.pt', weights_only=False)
        results = {}
        # find files containing model checkpoints
        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix == '.pt' and f.stem.isdigit()
        ]
        files.sort(key=lambda f: int(f.stem))

        for file in files:
            step = int(file.stem)
            print(f'Step {step}', flush=True)

            ddpm.load_state_dict(torch.load(f'{directory}/{step}.pt', weights_only=False))
            results[step/len(train_loader)] = sample_and_analyze_substrings(ddpm, args, train_loader, test_loader,
                                                                    eval_func)
        dict_progress['results'] = results
        torch.save(dict_progress, f'{directory}/test.pt')

    # training
    else:
        if args.resume_checkpoint:
            # load existing logs for resuming training
            dict_progress = torch.load(f'{directory}/logs.pt', weights_only=False)
            log_results = dict_progress['results']
            # find largest step that has finished a complete epoch
            epoch = max((k for k in log_results.keys() if isinstance(k, float) and k.is_integer()), default=None)
            step = log_results[epoch]['step']
            ddpm.load_state_dict(torch.load(f'{directory}/{step}.pt', weights_only=False))
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print('Created directory')
            dict_progress = None


        if args.grid:
            run_with_grid(train(train_loader, test_loader, ddpm, optim_sched, args, eval_func, resume_dict=dict_progress,
                                rule_freqs=rule_freqs, logprob=total_logprob_per_sample))
        else:
            for data in train(train_loader, test_loader, ddpm, optim_sched, args, eval_func, resume_dict=dict_progress,
                              rule_freqs=rule_freqs, logprob=total_logprob_per_sample): pass



