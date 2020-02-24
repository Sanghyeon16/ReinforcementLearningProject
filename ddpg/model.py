
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hiddens=[400,300]):
        super(Actor, self).__init__()

        dims = [nb_states] + hiddens + [nb_actions]
        self.linears = nn.ModuleList([nn.Linear(n1, n2) for n1, n2 in zip(dims[:-1], dims[1:])])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        out = self.linears[0](x)
        for linear in self.linears[1:]:
            out = self.relu(out)
            out = linear(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hiddens=[400,300]):
        super(Critic, self).__init__()
        dims = [nb_states + nb_actions] + hiddens + [1]
        self.linears = nn.ModuleList([nn.Linear(n1, n2) for n1, n2 in zip(dims[:-1], dims[1:])])
        self.relu = nn.ReLU()
    
    def forward(self, x, a):
        out = torch.cat([x, a], 1)
        out = self.linears[0](out)
        for linear in self.linears[1:]:
            out = self.relu(out)
            out = linear(out)
        return out
