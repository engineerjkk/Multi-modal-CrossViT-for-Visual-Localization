from .learning import LEARNINGS
from .checkpoints import Checkpoints
from .network import initialize_network


def initialize_learning(params, data, device):
    return LEARNINGS[params["learning"]["type"]].initialize(params, data, device)

def load_network(params, device):
    state = Checkpoints.load_network(params["path"])
    if state is not None:
        return initialize_network(None, device, state, params["runtime"])
    return initialize_network(params, device)

