from utils import *
from torch.utils.data import Dataset
import torch
from pomegranate.bayesian_network import BayesianNetwork
from pomegranate.distributions import ConditionalCategorical
from pomegranate.distributions import Categorical
from data import *
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm, trange
import random


def generate_data_to_csv(num_example=100):

    # num_example
    data = []
    mu = 0.1
    for i in trange(1500):

        if i > 250:
            mu = 0.2
        if i > 500:
            mu = 0.3

        if i > 1000:
            mu = 0.1
        if i > 1100:
            mu = 0.2
        if i > 1200:
            mu = 0.3
        if i > 1400:
            mu = 0.4

        graph_data = []
        graph = get_wet_grass_network(mu)
        samples = graph.sample(10000)

        base_vector = [0, 0]
        for data in samples:
            example = []
            for e in data:
                v = base_vector.copy()
                v[e] = 1
                example.append(v)
            graph_data.append(example)
        graph_data = np.array(graph_data)
        filename = f"wet_grass_data/graph{str(i)}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(graph_data, f)


class InContextDataset(Dataset):
    def __init__(self, N, num_example, train=True, mode="easy"):
        super().__init__()
        """
        N: number of samples
        num_example: number of examples in one input
        mode: easy -> graph_idx (0 ~ 249)
            : medium -> graph_idx (0 ~ 499)
            : hard -> graph_idx (0 ~ 999)
        """

        self.N = N
        self.num_example = num_example
        self.train = train
        self.graphs = []
        self.mode = mode
        if self.train:
            # graph index : [0 ~ 999]
            self.idx_start = 0
            if self.mode == "easy":
                self.idx_end = 249
            elif self.mode == "medium":
                self.idx_end = 499
            else:
                self.idx_end = 999

            self.fix = False

        else:
            # graph index : [1000 ~ 1499]
            self.fix = True
            self.idx_start = 1000
            self.idx_end = 1499

        for i in range(1500):
            self.graphs.append(self._load_data(i))
        if self.fix is True:
            self.fix = False
            self.data = []
            self.raw_data = []
            for j in range(self.N):
                d, r_d = self.__getitem__(1, prepare=True)
                self.data.append(d)
                self.raw_data.append(r_d)

            self.fix = True

    def update_mode(self, mode):
        self.mode = mode
        if self.mode == "easy":
            self.idx_end = 249
        elif self.mode == "medium":
            self.idx_end = 499
        else:
            self.idx_end = 999

    def _sampler_function(self, idx):
        data_source = torch.utils.data.TensorDataset(self._load_data(idx))
        sampler = torch.utils.data.RandomSampler(
            data_source, num_samples=(self.num_example + 1))

    def _load_data(self, idx):

        filename = f"wet_grass_data/graph{idx}.pkl"
        with open(filename, 'rb') as f:
            x = pickle.load(f)
        return torch.from_numpy(x)

    def _sample(self):

        graph_idx = random.randint(self.idx_start, self.idx_end)
        p = torch.ones(self.graphs[0].size(0)) / len(self.graphs[0])
        index = p.multinomial(
            num_samples=(
                self.num_example + 1),
            replacement=True)
        return self.graphs[graph_idx][index, :, :]

    def __len__(self):
        return self.N

    def __getitem__(self, idx, prepare=False):
        """
        This is for dynamic variable prediction
        y_idx = random.randint(0, 3)
        y = x[-1, y_idx].argmax().clone().detach()
        x[-1, y_idx, 0] = 0 # torch.zeros_like(x[-1, -1])
        x[-1, y_idx, 1] = 0 # torch.zeros_like(x[-1, -1])
        pos = torch.tensor([0, 0, 0, 0]).unsqueeze(0).repeat(x.size(0), 1)
        pos[-1, y_idx] = 1

        """
        if self.fix:
            return self.data[idx]
        else:
            if prepare is False:
                x = self._sample()  # (11, 4, 2)
                y_idx = random.randint(0, 3)
                y = x[-1, y_idx].argmax().clone().detach()
                x[-1, y_idx, 0] = 0  # torch.zeros_like(x[-1, -1])
                x[-1, y_idx, 1] = 0  # torch.zeros_like(x[-1, -1])
                pos = torch.tensor([0, 0, 0, 0]).unsqueeze(
                    0).repeat(x.size(0), 1)
                pos[-1, y_idx] = 1

                x = x.view(x.size(0), -1)
                x = torch.cat((x, pos), dim=-1)
                return x.float(), y.long()

            else:

                x = self._sample()  # (11, 4, 2)
                y_idx = random.randint(0, 3)
                y = x[-1, y_idx].argmax().clone().detach()

                # raw data for bayesian inference
                mask = torch.tensor([True, True, True, True])
                # True means the value is observed and False means that it is
                # not observed.
                mask[y_idx] = False
                raw_x = x.argmax(-1)
                raw_data = [raw_x, y, mask]

                y = x[-1, y_idx].argmax().clone().detach()
                x[-1, y_idx, 0] = 0  # torch.zeros_like(x[-1, -1])
                x[-1, y_idx, 1] = 0  # torch.zeros_like(x[-1, -1])
                pos = torch.tensor([0, 0, 0, 0]).unsqueeze(
                    0).repeat(x.size(0), 1)
                pos[-1, y_idx] = 1
                x = x.view(x.size(0), -1)
                x = torch.cat((x, pos), dim=-1)
                return (x.float(), y.long()), raw_data
