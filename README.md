# Transformer-Bayesian-Inference


### Installation
```
python -m venv ./env
source ./env/bin/activate
pip install -r requirements.txt
```

### Data  Sampling

```python
from utils import *

network = get_wet_grass_network() 
# This will generate a Bayesian Network (4 variables) with randomly initialized probability table

x = network.sample(10) 
# This will sample from the network 10 times, resulting a (10, 4) matrix.

```

### In-Context Data Generation

```python

from data import DataGenerator

D = DataGenerator("wet_grass")

D.generate_in_context_examples(10) 
# This will initialize a wet_grass network, and generate 1 in-context sample with 10 examples.
# All these 10 examples are sampled from the same network, and encoded in the way that paper describes

```