from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch
import numpy as np

def get_prob_dist_binary():

    p = np.random.uniform(0,1)
    return [p, 1-p]


def get_wet_grass_network():

    cloudy = Categorical([get_prob_dist_binary()])
    rain = ConditionalCategorical([[get_prob_dist_binary(), get_prob_dist_binary()]])
    sprinkler = ConditionalCategorical([[get_prob_dist_binary(), get_prob_dist_binary()]])
    wet_grass = ConditionalCategorical([[[get_prob_dist_binary(), get_prob_dist_binary()], [get_prob_dist_binary(), get_prob_dist_binary()]]])

    model = BayesianNetwork()
    model.add_distributions([cloudy, rain, sprinkler, wet_grass])
    model.add_edge(cloudy, rain)
    model.add_edge(cloudy, sprinkler)

    model.add_edge(rain, wet_grass)
    model.add_edge(sprinkler, wet_grass)

    return model 
