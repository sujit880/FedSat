import torch
from torch.optim import Optimizer
from torch.autograd import Variable

class PerturbedGradientDescent(Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""
    def __init__(self, params, learning_rate=0.001, mu=0.01):
        defaults = dict(learning_rate=learning_rate, mu=mu)
        super(PerturbedGradientDescent, self).__init__(params, defaults)
        for group in self.param_groups:
            # print('group:', group)
            for p in group['params']:
                state = self.state[p]
                # print("p :", p)
                # State initialization
                if 'vstar' not in state:
                    state['vstar'] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super(PerturbedGradientDescent, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['vstar'] = torch.zeros_like(p.data)

                vstar = state['vstar']
                w_diff = torch.pow(torch.norm(p.data - vstar.data), 2)
                # print(f'P: {p.data}, \nV: {vstar.data}')

                lr_t = group['learning_rate']
                mu_t = group['mu']

                # p.data.add_(-lr_t, grad + mu_t * (p.data - vstar))
                p.data.add_(grad + mu_t * (w_diff/2), alpha=-lr_t)

        return loss

    def set_params(self, cog):
        with torch.no_grad():
            # print("Start Setting")
            # starting = 0 
            for group in self.param_groups:
                # print("Old:", group['params'], "\nNew:", g_grad)
                # starting +=1
                # print(f'group count: {starting}')
                for p, grad in zip(group['params'], cog):
                    # print(f'P: {len(p.data)}, Grad: {len(grad)}')
                    self.state[p]['vstar'].copy_(torch.from_numpy(grad))
                #     self.state[p]['vstars'].copy_(grad)

