from models importn *
import argparse
from data import *
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Training Transformers for Bayesian Inference')

# data args
parser.add_argument('--N', help='Number of training samples', required=True, default=5000)
parser.add_argument('--num_example', help='Number of examples for in-context learning', required=True, default=100)

# training args
parser.add_argument('--epoch', help='Number of training epoch', required=True, default=200)
parser.add_argument('--init_lr', help='Initial learning rate', required=True, default=0.001)

# model args
parser.add_argument('--layers', help='Number of transformer layers', required=True, default=6)
parser.add_argument('--heads', help='Number of transformer attention heads', required=True, default=4)
parser.add_argument('--hid_dim', help='Size of hidden dimension', required=True, default=128)


args = vars(parser.parse_args())


def train_step(model, xs, ys, optimizer, loss_func):

    optimizer.zero_grad()
    output = model(xs)

    print(output.size()) # has to be a 1D thing
    raise Exception

    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def eval_step(model, xs, ys, loss_func):

    with torch.no_grad():
        output = model(xs)

        print(output.size()) # has to be a 1D thing
        raise Exception

        loss = loss_func(output, ys)
        loss.backward()
        optimizer.step()

    return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def train(args):

    model = TransformerModel(12, n_position=args.num_example+1, n_embd=args.hid_dim, n_layer=args.layers, n_head=args.heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)

    trainset = InContextDataset(args.N, args.num_example)
    testset = InContextDataset(args.N, args.num_example)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    loss_func = torch.nn.BCEWithLogitsLoss()
    metric = BinaryAccuracy()

    log = {
        "Training Loss":[],
        "Training Acc.":[],
        "Test Loss":[],
        "Test Acc.":[],
        "Epoch":[]
    }

    for epoch in args.epoch:

        train_loss = []
        train_acc = []

        test_loss = []
        test_acc = []

        model.train()
        for x, y in train_loader:

            loss, output = train_step(model, x, y, optimizer, loss_func)

            prediction = torch.sigmoid(output) > 0.5
            accuracy = metric(prediction, y)

            train_loss.append(loss)
            train_acc.append(accuracy.detach().item())

        model.eval()
        for x, y in test_loader:

            loss, output = eval_step(model, x, y, loss_func)

            prediction = torch.sigmoid(output) > 0.5
            accuracy = metric(prediction, y)

            test_loss.append(loss)
            test_acc.append(accuracy.detach().item())

        log["Training Loss"].append(np.mean(train_loss))
        log["Training Acc."].append(np.mean(train_acc))

        log["Test Loss"].append(np.mean(test_loss))
        log["Test Acc."].append(np.mean(test_acc))

        log["Epoch"].append(epoch)

    
    df = pd.DataFrame(log)

    fig, axes = plt.subplots(1, 2)

    sns.lineplot(data=df, x="Epoch", y="Training Loss", ax=axes[0])
    sns.lineplot(data=df, x="Epoch", y="Test Loss", ax=axes[0])

    sns.lineplot(data=df, x="Epoch", y="Training Acc.", ax=axes[1])
    sns.lineplot(data=df, x="Epoch", y="Test Acc.", ax=axes[1])

    plt.title(f"Training Data Size ({args.N}), ICL Examples ({args.num_example})")
    plt.legend()
    plt.tight_layout()

    plt.savefig( f"{args.N}_{args.num_example}_{args.layers}_{args.hid_dim}_{args.heads}.png", dpi=600)