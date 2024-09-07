from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch

import numpy as np


def get_prob_dist_binary():

    p = np.random.uniform(0, 1)
    return [p, 1 - p]


def simple_DAG():

    A = Categorical([[0.5, 0.5]])
    B = ConditionalCategorical([[[0.5, 0.5], [0.5, 0.5]]])
    # sprinkler = ConditionalCategorical([[get_prob_dist_binary(), get_prob_dist_binary()]])
    # wet_grass = ConditionalCategorical([[[get_prob_dist_binary(), get_prob_dist_binary()], [get_prob_dist_binary(), get_prob_dist_binary()]]])

    model = BayesianNetwork()
    model.add_distributions([A, B])
    model.add_edge(A, B)
    return model


def generate_wet_grass_data():

    # cloudy_p = get_prob_dist_binary()
    # rain_p1
    # rain_p2
    # sprinkler_p1
    # sprinkler_p2
    # wet_grass_p1
    # wet_grass_p2
    # wet_grass_p3
    # wet_grass_p4

    cloudy = Categorical([get_prob_dist_binary()])
    rain = ConditionalCategorical(
        [[get_prob_dist_binary(), get_prob_dist_binary()]])
    sprinkler = ConditionalCategorical(
        [[get_prob_dist_binary(), get_prob_dist_binary()]])
    wet_grass = ConditionalCategorical([[[get_prob_dist_binary(), get_prob_dist_binary()], [
                                       get_prob_dist_binary(), get_prob_dist_binary()]]])

    model = BayesianNetwork()
    model.add_distributions([cloudy, rain, sprinkler, wet_grass])
    model.add_edge(cloudy, rain)
    model.add_edge(cloudy, sprinkler)

    model.add_edge(rain, wet_grass)
    model.add_edge(sprinkler, wet_grass)

    return model


# model = generate_wet_grass_data()
# out = model.sample(100)
# # print(out)
# # model.fit(out)

# x = model.sample(10)
# print(x)
# print(model.probability(x))

# model = simple_DAG()
# out = model.sample(100)
# # print(out)
# # model.fit(out)

# x = model.sample(10)
# print(x)
# print(model.probability(x))
