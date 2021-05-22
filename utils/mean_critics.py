import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class MeanCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(MeanCritic, self).__init__()

        self.sa_sizes = sa_sizes # 列表，有10个元素，每个都是 (22,5)
        self.nagents = len(sa_sizes)
        odim = 5

        self.mean_embedding = nn.Sequential()
        self.mean_embedding.add_module('critic_me_fc1', nn.Linear(2, 64)) # relative position
        self.mean_embedding.add_module('critic_me_nl', nn.LeakyReLU())

        self.critics = nn.Sequential()
        self.critics.add_module('critic_fc1', nn.Linear(75, hidden_dim))
        self.critics.add_module('critic_nl', nn.LeakyReLU())
        self.critics.add_module('critic_fc2', nn.Linear(hidden_dim, odim))



    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.sa_sizes))
        states = [s for s, a in inps] # (10, 1024, 22)
        actions = [a for s, a in inps] # (10, 1024, 5)


        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            obset = []
            for j in range(0, states[i].shape[1], 2):
                obset.append(states[i][:,j:j+2]) #  (11, 1024, 2)

            me_state = []
            for j in range(3, len(obset)):
                me_state.append(self.mean_embedding(obset[j])) # (9, 1024, 2)

            me_out = torch.zeros_like(me_state[0], dtype=torch.float32)
            for j in me_state:
                me_out += j
            me_out /= len(me_state) # (1024, 64)

            critic_in_state = torch.cat((obset[0], obset[1], obset[2], me_out), dim=1) # (1024, 68)

            agent_act = actions[i] # (1024, 5)
            other_act = [actions[j] for j in range(len(states)) if j is not i] # (9, 1024, 5)
            mean_act = torch.zeros_like(other_act[0], dtype=torch.float32)
            for j in other_act:
                mean_act += j
            mean_act /= len(mean_act) # (1024, 5)

            critic_in = torch.cat((critic_in_state, mean_act), dim=1) # (1024, 75)
            # 对智能体i, 输入状态为自己的位置速度(4)，地标状态(2)其他智能体的相对位置编码(64)，其他智能体均值场动作(5)

            all_q = self.critics(critic_in) # (1024, 5)


            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)


            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
