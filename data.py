from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork

import torch
from torch.utils.data import Dataset

import numpy as np
from utils import *
import random


class InContextDataset(Dataset):
    def __init__(self, N, num_example, n_graphs=4):
        super().__init__()
        """
        N: number of samples
        num_example: number of examples in one input
        """

        self.N = N
        self.num_example = num_example
        self.generator = DataGenerator("wet_grass", n_graphs)
        self.data = self._generate_data()

        # Just generate num_example + 1 in-context, and then mask out the last one.
        # Create positional embedding
        # Ground Truth is label for binary classification

    def _generate_data(self):

        data = []
        for _ in range(self.N):
            example = torch.tensor(
                self.generator.generate_in_context_examples(
                    self.num_example + 1))
            data.append(example)  # include the test token
        return data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        x = self.data[idx]  # (11, 4, 2)
        y = x[-1, -1].argmax().clone().detach()
        x[-1, -1, 0] = 0  # torch.zeros_like(x[-1, -1])
        x[-1, -1, 1] = 0  # torch.zeros_like(x[-1, -1])

        # pos = torch.zeros(x.size(0), int(x.size(1)))
        # pos[-1, -1] = 1
        pos = torch.tensor([0, 0, 0, 0]).unsqueeze(0).repeat(x.size(0), 1)
        pos[-1, -1] = 1
        x = x.view(x.size(0), -1)
        x = torch.cat((x, pos), dim=-1)
        return x.float(), y.long()


class DataGenerator:
    def __init__(self, network_type, n_graphs=4):
        self.network_type = network_type
        self.n_graphs = n_graphs

        if self.network_type == "wet_grass":
            self.generation_fn = get_wet_grass_network
            self.graph_m = 4  # the number of variables in the graph

        elif self.network_type == "Toy":
            self.generation_fn = simple_DAG
            self.graph_m = 2  # the number of variables in the graph

        self.graphs = self._initialize_networks()

    def _initialize_networks(self):
        return [self.generation_fn() for _ in range(self.n_graphs)]

    def generate_in_context_examples(self, n):
        """
        n: the number of in-context examples per input
        """

        result = []
        idx = random.randint(0, self.n_graphs - 1)

        graph = self.graphs[idx]
        data_gen = graph.sample(n)  # (n, graph_m)
        base_vector = [0, 0]
        for data in data_gen:
            example = []
            for e in data:
                v = base_vector.copy()
                v[e] = 1
                example.append(v)

            result.append(example)

        return result
