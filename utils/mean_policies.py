import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x

        # state embedding
        self.mean_embedding = nn.Sequential()
        self.mean_embedding.add_module('me_fc1', nn.Linear(2, 64)) # relative position
        self.mean_embedding.add_module('me_nl', nn.LeakyReLU())

        # policy net
        self.p_net = nn.Sequential()
        self.p_net.add_module('p_fc1', nn.Linear(68, hidden_dim)) # 128
        self.p_net.add_module('p_nl1', nn.LeakyReLU())
        self.p_net.add_module('p_fc2', nn.Linear(hidden_dim, hidden_dim))
        self.p_net.add_module('p_nl2', nn.LeakyReLU())
        self.p_net.add_module('p_fc3', nn.Linear(hidden_dim, out_dim))

        # self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, out_dim)
        # self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        # X: (3, 22)
        # X.shape[0] = 3, X.shape[1] = 22

        X = self.in_fn(X) # batch_norm
        obset = []
        for i in range(0, X.shape[1], 2):
            obset.append(X[:,i:i+2]) #  (11, 3, 2)

        me_state = []
        for i in range(2, len(obset)):
            me_state.append(self.mean_embedding(obset[i])) # (9, 3, 2)

        me_out = torch.zeros_like(me_state[0], dtype=torch.float32)
        for i in me_state:
            me_out += i
        me_out /= len(me_state) # (3, 64)

        pol_in = torch.cat((obset[0], obset[1], me_out), dim=-1) # (3, 68)
        out = self.p_net(pol_in)
        return out


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0] # 生成与环境交互采样数据
        return rets
