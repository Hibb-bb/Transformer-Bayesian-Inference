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
from utils import *

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
parser.add_argument(
    '--ckpt',
    help='checkpoint path',
    type=str,
    default='./ckpt/wet_grass')


# log args
parser.add_argument(
    '--log-every',
    help='log every X steps',
    type=int,
    default=100)

args = parser.parse_args()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(torch.randint(0, total_seeds - 1))
    return seeds


def test_epoch(model, test_loader, metric, args):

    with torch.no_grad():

        model.eval()
        num_correct_test = 0
        for x, y in test_loader:

            x, y = x.cuda(), y.cuda()
            output = model(x)
            prediction = torch.softmax(output.detach(), dim=-1).argmax(-1)
            num_correct_test += int(metric(prediction, y) * y.size(0))

            torch.cuda.empty_cache()

    return num_correct_test / 15000


def bayesian_inference(test_loader, args):

    with torch.no_grad():

        num_correct_test = 0
        for i in trange(len(test_loader.raw_data)):

            (X, y, mask) = test_loader.raw_data[i]
            network = get_wet_grass_network()
            X_in = X[:-1]
            X_test = X[-1]
            network.fit(X_in)
            X_masked = torch.masked.MaskedTensor(
                X_test, mask=mask).unsqueeze(0)
            prediction = network.predict(X_masked)
            if prediction[0][-1] == X_test[-1]:
                num_correct_test += 1

    return num_correct_test / 15000


def run(args):

    log = {
        "Test Acc.": [],
        "Step": []
    }
    testset = InContextDataset(15000, args.num_example, train=False)
    test_loader = DataLoader(testset, batch_size=512, shuffle=False)

    print("Preparing model...")
    model = TransformerModel(
        12,
        n_positions=args.num_example + 1,
        n_embd=args.hid_dim,
        n_layer=args.layers,
        n_head=args.heads)
    model = model.cuda()
    metric = BinaryAccuracy().cuda()
    log_path = f"ckpt/wet_grass/{args.N}_{args.num_example}_{args.layers}_{args.hid_dim}_{args.heads}_{args.batch_size}_{args.steps}/"
    bayesian_acc = bayesian_inference(testset, args)
    print(bayesian_acc)
    for i in trange(args.steps):
        if i != 0 and i % args.log_every == 0:

            ckpt_path = f"{log_path}/step_{i}.pt"
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            model.eval()
            test_acc = test_epoch(model, test_loader, metric, args)

            log["Test Acc."].append(test_acc)
            log["Step"].append(i)

    log["Bayesian Inference Acc."] = [bayesian_acc] * len(log["Step"])
    df = pd.DataFrame(log)

    sns.lineplot(
        data=df,
        x="Step",
        y="Bayesian Inference Acc.",
        color="orange",
        alpha=0.8,
        label="Bayesian Inference")
    sns.lineplot(
        data=df,
        x="Step",
        y="Test Acc.",
        color="lightblue",
        alpha=0.8,
        label="Transformer")
    plt.legend()
    plt.title("Transformer v.s. Bayesian Inference")
    plt.tight_layout()
    plt.savefig(f"{log_path}/evaluation.png")


run(args)
