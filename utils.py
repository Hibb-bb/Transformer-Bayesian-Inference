from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch
import numpy as np


def get_prob_dist_binary(mu_prime=0.1):

    eps = 1e-3

    # Parameters for the two Gaussian distributions
    mean1, std1, weight1 = (0 + mu_prime), 1.0, 0.5
    mean2, std2, weight2 = (1 - mu_prime), 1.0, 0.5

    num_samples = 1

    samples1 = torch.normal(mean1, std1, size=(num_samples,)) * weight1
    samples2 = torch.normal(mean2, std2, size=(num_samples,)) * weight2

    p = samples1 + samples2
    p = torch.clamp(p, 0, 1)

    if p == 1:
        p = p - eps
    elif p == 0:
        p = p + eps

    return [p, 1 - p]

# def get_prob_dist_binary():

#     p = np.random.uniform(0, 1)
#     return [p, 1 - p]


def get_wet_grass_network(mu=0.1):

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
