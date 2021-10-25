from functools import partial
from math import sqrt
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class HyperLinear(nn.Module):
    """
    Linear network layers that allows for two additional complications:
        - parameters admit to be connected via a hyper-network like structure
        - network weights are transformed according to some rule before application
    """

    def __init__(self, in_size, out_size, use_hypernetwork=True):
        super(HyperLinear, self).__init__()

        self.use_hypernetwork = use_hypernetwork

        if not self.use_hypernetwork:
            self.w = nn.Linear(in_size, out_size)
        self.b = nn.Parameter(th.randn(out_size))

        # initialize layers
        stdv = 1. / sqrt(in_size)
        if not self.use_hypernetwork:
            self.w.weight.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

        pass

    def forward(self, inputs, weights=None, weight_mod="abs", hypernet=None, **kwargs):
        """
        we assume inputs are of shape [a*bs*t]*v
        """
        assert inputs.dim() == 2, "we require inputs to be of shape [a*bs*t]*v"

        if self.use_hypernetwork:
            assert weights is not None, "if using hyper-network, need to supply the weights!"
            w = weights
        else:
            w = self.w.weight

        weight_mod_fn = None
        if weight_mod in ["abs"]:
            weight_mod_fn = th.abs
        elif weight_mod in ["pow"]:
            exponent = kwargs.get("exponent", 2)
            weight_mod_fn = partial(th.pow, exponent=exponent)
        elif callable(weight_mod):
            weight_mod_fn = weight_mod

        if weight_mod_fn is not None:
            w = weight_mod_fn(w)

        x = th.mm(inputs, w.t()) + self.b # TODO: why not BMM?
        return x


class CEMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CEMAgent, self).__init__()
        self.args = args
        num_inputs = input_shape + args.n_actions
        hidden_size = args.rnn_hidden_dim

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def get_weight_decay_weights(self):
        return {}

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions):
        if actions is not None:
            inputs = th.cat([inputs, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return {"Q":q, "hidden_state": x}


class CEMRecurrentAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CEMRecurrentAgent, self).__init__()
        self.args = args
        num_inputs = input_shape + args.n_actions
        hidden_size = args.rnn_hidden_dim

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def get_weight_decay_weights(self):
        return {} # TODO: implement

    def forward(self, inputs, hidden_state, actions):
        if actions is not None:
            inputs = th.cat([inputs, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return {"Q":q, "hidden_state":h}