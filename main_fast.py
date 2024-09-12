import argparse
from tqdm import trange, tqdm

from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
from generate_data import *
from models import *

from pathlib import Path

parser = argparse.ArgumentParser(
    description='Training Transformers for Bayesian Inference')

# data args
parser.add_argument(
    '--N',
    help='Number of training samples',
    type=int,
    default=10000)
parser.add_argument(
    '--num-example',
    help='Number of examples for in-context learning',
    type=int,
    default=100)

# training args
parser.add_argument(
    '--steps',
    help='Number of training steps',
    type=int,
    default=50000)
parser.add_argument(
    '--init_lr',
    help='Initial learning rate',
    type=float,
    default=1e-4)
parser.add_argument('--batch-size', help='Batch size', type=int, default=64)

# model args
parser.add_argument(
    '--layers',
    help='Number of transformer layers',
    type=int,
    default=12)
parser.add_argument(
    '--heads',
    help='Number of transformer attention heads',
    type=int,
    default=4)
parser.add_argument(
    '--hid_dim',
    help='Size of hidden dimension',
    type=int,
    default=128)

# log args
parser.add_argument(
    '--log-every',
    help='log every X steps',
    type=int,
    default=1000)

args = parser.parse_args()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(torch.randint(0, total_seeds - 1))
    return seeds


def train_step(model, optimizer, x, y, loss_func, metric, args):

    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()
    output = model(x)
    loss = loss_func(output, y)
    loss.backward()
    optimizer.step()

    prediction = torch.softmax(output.detach(), dim=1).argmax(-1)
    acc = metric(prediction, y)

    torch.cuda.empty_cache()

    return acc.item(), loss.item()


def test_epoch(model, loss_func, test_loader, metric, args):

    test_loss = []
    with torch.no_grad():

        model.eval()
        num_correct_test = 0
        for x, y in test_loader:

            x, y = x.cuda(), y.cuda()
            output = model(x)
            test_loss_ = loss_func(output, y)
            prediction = torch.softmax(output.detach(), dim=-1).argmax(-1)
            test_loss.append(test_loss_.item())
            num_correct_test += int(metric(prediction, y) * y.size(0))

            torch.cuda.empty_cache()

    return num_correct_test / 20000, np.mean(test_loss)


def train(args):

    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="Transformers-Bayesian-Inference",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": args.init_lr,
    #         "epochs": args.epoch,
    #         "hid_dim": args.hid_dim,
    #         "heads": args.heads,
    #         "layers": args.layers,
    #         "N": args.N,
    #         "num_example": args.num_example,
    #     },
    # )

    trainset = InContextDataset(
        args.steps *
        args.batch_size,
        args.num_example,
        train=True,
        mode="easy")
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True)

    testset = InContextDataset(20000, args.num_example, train=False)
    test_loader = DataLoader(testset, batch_size=512, shuffle=False)

    print("Preparing model...")
    model = TransformerModel(
        12,
        n_positions=args.num_example + 1,
        n_embd=args.hid_dim,
        n_layer=args.layers,
        n_head=args.heads)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)

    print("Preparing metrics...")
    loss_func = torch.nn.CrossEntropyLoss()
    metric = BinaryAccuracy().cuda()

    log = {
        "Training Loss": [],
        "Training Acc.": [],
        "Test Loss": [],
        "Test Acc.": [],
        "Step": []
    }

    print("Start Training...")
    log_path = f"ckpt/wet_grass/{args.N}_{args.num_example}_{args.layers}_{args.hid_dim}_{args.heads}_{args.batch_size}_{args.steps}/"
    path = Path(log_path)
    path.mkdir(parents=True, exist_ok=True)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    num_correct_train = 0

    p_bar = trange(args.steps)

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps)
    log_path_txt = log_path + "log.txt"
    f = open(log_path_txt, "w")

    for step in p_bar:

        _i, (x, y) = next(enumerate(train_loader))

        model.train()
        train_acc_step, train_loss_step = train_step(
            model, optimizer, x, y, loss_func, metric, args)
        train_loss.append(train_loss_step)
        train_acc.append(train_acc_step)

        if step != 0 and step % 1000 == 0:
            sch.step()

        if step == 10000:
            trainset.update_mode("medium")
            train_loader = DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True)

        if step == 20000:
            trainset.update_mode("hard")
            train_loader = DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True)

        if step != 0 and step % args.log_every == 0:

            ckpt_path = f"{log_path}/step_{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            test_model = TransformerModel(
                12,
                n_positions=args.num_example + 1,
                n_embd=args.hid_dim,
                n_layer=args.layers,
                n_head=args.heads)
            test_model = test_model.cuda()
            test_model.eval()
            test_model.load_state_dict(
                torch.load(ckpt_path, weights_only=True))

            test_acc_step, test_loss_step = test_epoch(
                test_model, loss_func, test_loader, metric, args)
            test_loss.append(test_loss_step)
            test_acc.append(test_acc_step)
            p_bar.set_description(
                f"| Step: {step} | Train Loss: {str(round(np.mean(train_loss), 4))} | Test Loss: {str(round(np.mean(test_loss), 4))} | Lr: {sch.get_last_lr()}")

            del test_model

            log["Training Loss"].append(np.mean(train_loss))
            log["Test Loss"].append(np.mean(test_loss))
            log["Training Acc."].append(np.mean(train_acc))
            log["Test Acc."].append(np.mean(test_acc))
            log["Step"].append(step)

            log_path_txt = log_path + "log.txt"
            f = open(log_path_txt, "a")
            f.write("|Step|" +
                    "\t" +
                    str(step) +
                    "\t" +
                    "|Train Loss:|" +
                    "\t" +
                    str(round(np.mean(train_loss), 4)) +
                    "\t" +
                    "|Train Acc:|" +
                    "\t" +
                    str(round(np.mean(train_acc), 4)) +
                    "\n")
            f.write("|Step|" +
                    "\t" +
                    str(step) +
                    "\t" +
                    "|Test Loss:|" +
                    "\t" +
                    str(round(np.mean(test_loss), 4)) +
                    "\t" +
                    "|Test Acc:|" +
                    "\t" +
                    str(round(np.mean(test_acc), 4)) +
                    "\n")
            f.write("\n")

            train_loss = []
            train_acc = []
            test_loss = []
            test_acc = []

            num_correct_train = 0

    df = pd.DataFrame(log)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.lineplot(
        data=df,
        x="Step",
        y="Training Loss",
        ax=axes[0],
        color="orange",
        alpha=0.8,
        label="Train")
    sns.lineplot(
        data=df,
        x="Step",
        y="Test Loss",
        ax=axes[0],
        color="lightblue",
        alpha=0.8,
        label="Test")
    axes[0].legend()

    sns.lineplot(
        data=df,
        x="Step",
        y="Training Acc.",
        ax=axes[1],
        color="orange",
        alpha=0.8,
        label="Train")
    sns.lineplot(
        data=df,
        x="Step",
        y="Test Acc.",
        ax=axes[1],
        color="lightblue",
        alpha=0.8,
        label="Test")
    axes[1].legend()

    plt.title(f"Steps ({args.steps}), ICL Examples ({args.num_example})")
    plt.tight_layout()
    plt.savefig(f"{log_path}/curve_plot.png")


train(args)
