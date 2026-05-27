import torch
import math
from time import time
from copy import deepcopy


def generate_epoch_checkpoints(args):
    max_exp = int(math.log2(args.n_epoch))
    num_checkpoints = max_exp * args.save_freq + 1
    times = torch.logspace(0, max_exp, num_checkpoints, base=2).int()
    times = times.tolist() + [args.n_epoch]
    return list(set(times))

def generate_step_checkpoints(args):
    max_steps = args.n_epoch * (args.train_size // args.batch_size + int( (args.train_size % args.batch_size) > 0))
    max_exp = int(math.log2(max_steps))
    num_checkpoints = max_exp * args.save_freq + 1
    times = torch.logspace(0, max_exp, num_checkpoints, base=2).int()
    times = times.tolist() + [max_steps]
    return list(set(times))

def evaluate_model(step, args, ddpm, model0, trainloader, testloader, eval_func, loss_ema=None, time_wall=None):
    ddpm.train()
    loss = 0.0
    num_points = 0
    for x, _ in trainloader:
        num_points += x.shape[0]
        x = x.to(args.device)
        loss += ddpm(x, args.n_trajectories).item() * len(x)
    loss /= num_points
    log_step = {}
    log_step["step"] = step
    log_step["loss"] = loss
    log_step["loss_ema"] = loss_ema
    log_step["Wall_time"] = time_wall

    ddpm.eval()
    with torch.no_grad():
        x = next(iter(trainloader))[0].to(args.device)
        x_test = next(iter(testloader))[0].to(args.device)
        xh = ddpm.sample(8192, (x.shape[1], x.shape[2]), args.device)

        for key, value in eval_func.items():
            if key == "Time_losses":
                val = value(ddpm, x)
            elif key in ["Test_losses", "True_losses"]:
                val = value(ddpm, x_test)
            elif key == "Weight_norm":
                val = value(ddpm.model, model0)
            else:
                val = value(xh)
                print(f"{key} : {val}", flush=True)
            log_step[key] = val

    return log_step


def print_epoch(i, time0, loss_ema, loss):
    time_wall = time()
    print(f"Epoch {i}, wall t {(time_wall-time0):.0f}s : loss_ema={loss_ema:.4f}; loss={loss:.4f}", flush=True)


def train(trainloader, testloader, ddpm, optim_sched, args, eval_func={}, resume_dict=None, rule_freqs=None, logprob=None):
    print(args, flush=True)

    epoch_checkpoints = generate_epoch_checkpoints(args)
    step_checkpoints = generate_step_checkpoints(args)
    optim, scheduler = optim_sched

    time0 = time()
    model0 = deepcopy(ddpm.model)

    if resume_dict is not None:
        log_results = resume_dict['results']
        epoch = max((k for k in log_results.keys() if isinstance(k, float) and k.is_integer()), default=None)
        step = log_results[epoch]['step']
        start_epoch = int(epoch + 1)
    else:
        log_results = {}
        step = 0
        log_results[0] = evaluate_model(step, args, ddpm, model0, trainloader, testloader, eval_func)
        print_epoch(0, time0, log_results[0]["loss"], loss=log_results[0]["loss"])
        start_epoch = 1

    for i in range(start_epoch, args.n_epoch + 1):

        ddpm.train()
        loss_ema = None
        for i_batch, (x, _) in enumerate(trainloader):
            optim.zero_grad()
            x = x.to(args.device)
            loss = ddpm(x, args.n_trajectories)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            optim.step()
            scheduler.step() # Step the learning rate scheduler
            step += 1
            current_epoch = (i-1) + (i_batch+1) / len(trainloader)

            if ((i_batch+1)==len(trainloader) and i in epoch_checkpoints) or (step in step_checkpoints):
                data = {}
                log_step = evaluate_model(
                                step, args, ddpm, model0, trainloader, testloader, eval_func, loss_ema=loss_ema, time_wall=time()-time0
                            )
                log_results[current_epoch] = log_step

                if not args.grid:
                    # save model
                    torch.save(
                        ddpm.state_dict(), f"./results/{args.output}_ddpm_{args.dataset}_zipf{args.zipf}_layer{args.layer}/{step}.pt"
                    )

                    # save logs
                    torch.save(
                        {"results": log_results, "args": args, "rule_freqs": rule_freqs, "logprob": logprob},
                        f"./results/{args.output}_ddpm_{args.dataset}_zipf{args.zipf}_layer{args.layer}/logs.pt",
                    )
                else:
                    data["args"] = args
                    data["results"] = log_results
                    data["ddpm_state"] = ddpm.state_dict()
                    if rule_freqs:
                        data["rule_freqs"] = rule_freqs
                    if logprob:
                        data["logprob"] = logprob

                yield data
            
        if i % args.print_period == 0:
            print_epoch(i, time0, loss_ema, loss=loss.item())
