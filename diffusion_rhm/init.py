from functools import partial

import torch

import datasets
from diffusion.ddpm import DDPM, DiscreteDDPM
from diffusion.unet import hUNET, hUNETTimeChan, hUNETFullEmb, bpUNET
from diffusion.evaluate_model import (
    check_generated_pixels,
    compute_loss_per_time,
    compute_d3pm_loss_per_time,
    cross_entropy_tuples,
    check_rules,
    compare_with_trainset,
    hierarhical_copies,
)
from belief_propagation.BP_utils import BpRhm


def init_data(args, trainloader_shuffle=True):
    """
    Initialise dataset.

    Returns:
        Two dataloaders for train and test set.
    """
    if args.dataset == "rhm":
        # create distribution for rules according to Zipf parameter
        probability = None

        if args.zipf_exponent is not None:
            assert args.zipf_layer is not None, "zipf law requires layer of application"
            probability = {}
            # initialize every layer to be uniformly distributed
            for l in range(args.num_layers):
                probability[l] = torch.ones(args.num_synonyms) / args.num_synonyms
            zipf_prob = torch.zeros(args.num_synonyms)
            # modify probabilities at layer with Zipf distribution
            for i in range(args.num_synonyms):
                zipf_prob[i] = (i + 1) ** (- float(args.zipf_exponent))
            probability[args.zipf_layer - 1] = zipf_prob / zipf_prob.sum()

        dataset = datasets.RandomHierarchyModel(
            num_features=args.num_features,  # vocabulary size
            num_synonyms=args.num_synonyms,  # features multiplicity
            num_layers=args.num_layers,  # number of layers
            num_classes=args.num_classes,  # number of classes
            tuple_size=args.tuple_size,  # number of branches of the tree
            seed_rules=args.seed_rules,
            train_size=args.train_size,
            test_size=args.test_size,
            seed_sample=args.seed_sample,
            input_format=args.input_format,
            whitening=args.whitening,
            replacement=args.replacement,
            probability=probability,
            unique=args.unique,
        )
        rules = dataset.get_rules()

    elif args.dataset == "single_pixel":

        dataset = datasets.SinglePixelModel(
            num_features=args.num_features,
            num_classes=args.num_classes,
            num_synonyms=args.num_synonyms,
            seed_rules=args.seed_rules,
            input_format=args.input_format,
            whitening=args.whitening,
        )
        rules = dataset.get_rules()

    else:
        raise ValueError("dataset argument is invalid!")

    dataset.features, dataset.labels = dataset.features.to(
        args.device
    ), dataset.labels.to(
        args.device
    )  # move to device when using cuda

    trainset = torch.utils.data.Subset(dataset, range(args.train_size))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=trainloader_shuffle, num_workers=0
    )

    if args.test_size:
        testset = torch.utils.data.Subset(
            dataset, range(args.train_size, args.train_size + args.test_size)
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=0
        )
    else:
        test_loader = None

    return train_loader, test_loader, rules, dataset.features, dataset.rule_freqs, dataset.total_logprob_per_sample


def init_model(args):
    """
    Initialise machine-learning model.
    """
    torch.manual_seed(args.seed_model)

    if args.model == "hUnet":
        assert args.filter_size is not None, "CNN model requires argument filter_size!"
        model = hUNET(
            input_dim=args.tuple_size**args.num_layers,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            width=args.width,
            num_layers=args.depth,
            bias=args.bias,
        )
    elif args.model == "hUnetTimeChan":
        assert args.filter_size is not None, "CNN model requires argument filter_size!"
        model = hUNETTimeChan(
            input_dim=args.tuple_size**args.num_layers,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            width=args.width,
            num_layers=args.depth,
            bias=args.bias,
        )
    elif args.model == "hUnetFullEmb":
        assert args.filter_size is not None, "CNN model requires argument filter_size!"
        model = hUNETFullEmb(
            input_dim=args.tuple_size**args.num_layers,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            width=args.width,
            num_layers=args.depth,
            bias=args.bias,
        )
    elif args.model == "bpUnet":
        model = bpUNET(
            input_dim=args.tuple_size**args.num_layers,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            width=args.width,
            num_layers=args.depth,
            bias=args.bias,
            process=args.process,
        )
    else:
        raise ValueError("model argument is invalid!")

    model = model.to(args.device)

    if args.process == "discrete":
        ddpm = DiscreteDDPM(
            model=model,
            betas=(args.beta1, args.beta2),
            n_T=args.nT,
            model_type=args.model_type,
            model_output=args.model_output,
        )
    else:
        ddpm = DDPM(
            model=model,
            betas=(args.beta1, args.beta2),
            n_T=args.nT,
            model_type=args.model_type,
            model_output=args.model_output,
        )

    return ddpm


class NoOpScheduler:
    def step(self):
        pass

def init_optimizer(model, args):
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError(f"Invalid optimizer {args.optim}")

    if args.warmup_steps > 0:
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = NoOpScheduler()

    return optimizer, scheduler


def init_eval_func(rules, bp, args, dataset):
    if args.dataset == "single_pixel":
        return {
            "Accuracy_and_frequencies": partial(
                check_generated_pixels, rules[0].to(args.device)
            ),
            "Time_losses": partial(compute_loss_per_time, 10, 1),
            "Cross-entropy_tuples": partial(
                cross_entropy_tuples, rules[args.num_layers - 1].to(args.device)
            ),
        }
    else:
        return {
            "Time_losses": partial(compute_d3pm_loss_per_time, 10, 1),
            "Test_losses": partial(compute_d3pm_loss_per_time, 10, 1),
            "True_losses": partial(compute_d3pm_loss_per_time, 10, 1, bp=bp),
            "Fraction_of_copies": partial(compare_with_trainset, dataset[: args.train_size]),
            "Hierarchical_copies": partial(hierarhical_copies, dataset=dataset[: args.train_size],
                                           num_layers=args.num_layers, tuple_size=args.tuple_size),
            "Valid_samples": partial(check_rules, bp=bp, trainset=dataset[: args.train_size]),
        }


def init_bp(args, rules):
    bp = BpRhm(
        args.num_features,
        args.tuple_size,
        args.num_synonyms,
        args.num_layers,
        rules,
        args.device,
    )
    return bp
