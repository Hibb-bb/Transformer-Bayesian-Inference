from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch

import numpy as np

from utils import *



class DataGenerator:
    def __init__(self, network_type):
        self.network_type = network_type
        if self.network_type == "wet_grass":
            self.generation_fn = get_wet_grass_network
            self.graph_m = 4 # the number of variables in the graph

        elif self.network_type == "Toy":
            self.generation_fn = simple_DAG
            self.graph_m = 2 # the number of variables in the graph

    def _initialize_networks(self):
        return [get_wet_grass_network() for _ in range(self.network_num)]

    def generate_in_context_examples(self, n):
        
        """
        n: the number of in-context examples per input
        """

        result = []
        graph = self.generation_fn()
        data_gen = graph.sample(n) # (n, graph_m)
        base_vector = [0, 0]
        for data in data_gen:
            example = []
            for e in data:
                v = base_vector.copy()
                v[e] = 1
                example.append(v)

            result.append(example)

        print(len(result), len(result[0]), result[0])
        print(data_gen[0])

D = DataGenerator("wet_grass")

D.generate_in_context_examples(4)