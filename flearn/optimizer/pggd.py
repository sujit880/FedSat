import torch
from torch.optim import Optimizer
from torch.autograd import Variable
import numpy as np


class PerGodGradientDescent(Optimizer):
    """Implementation of Perturbed gold Gradient Descent, i.e., FedDane optimizer"""

    def __init__(self, params, learning_rate=0.001, mu=0.01):
        defaults = dict(learning_rate=learning_rate, mu=mu)
        super(PerGodGradientDescent, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # State initialization
                if 'vstar' not in state:
                    state['vstar'] = torch.zeros_like(p.data)
                if 'gold' not in state:
                    state['gold'] = torch.zeros_like(p.data)

        # print('State', self.state)


    def __setstate__(self, state):
        super(PerGodGradientDescent, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        # print(f'Starting steps in groups bellow:')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if 'vstar' not in state:
                    state['vstar'] = torch.zeros_like(p.data)
                if 'gold' not in state:
                    state['gold'] = torch.zeros_like(p.data)

                vstar = state['vstar']
                gold = state['gold']

                # print(f'Vstar: {vstar}, Gold: {gold}')

                # w_diff = torch.pow(torch.norm(p.data - vstar.data), 2)

                lr_t = group['learning_rate']
                mu_t = group['mu']

                p.data.add_(grad + gold + mu_t * (p.data - vstar), alpha=-lr_t)
                # print(f'_Vstar: {self.state[p]["vstar"]}, _Gold: {self.state[p]["gold"]}')
                # p.data.add_(grad + gold + mu_t * (w_diff/2), alpha=-lr_t)

        return loss

    def set_params(self, cog, avg_gradient, client):
        for group in self.param_groups:
                for p, grad in zip(group['params'], cog):
                    self.state[p]['vstar'].copy_(torch.from_numpy(grad))

        _, gprev = client.get_grads()

        # Find g_t - F'(old)
        # gdiff = [torch.from_numpy(np.array(g1) - np.array(g2)) for g1, g2 in zip(avg_gradient, gprev)]
        gdiff = [torch.tensor(g1 - g2) for g1, g2 in zip(avg_gradient, gprev)]

        for group, grad in zip(list(self.param_groups), gdiff):
            for p in group['params']:
                self.state[p]['gold'].copy_(grad)
