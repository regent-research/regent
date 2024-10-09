# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D


class ActionDistribution(nn.Module):
    def __init__(self, ac_dim, ac_chunk=1):
        super().__init__()
        self._ac_chunk, self._ac_dim = ac_chunk, ac_dim
    
    @property
    def ac_dim(self):
        return self._ac_dim
    
    @property
    def ac_chunk(self):
        return self._ac_chunk
    
    @property
    def num_ac_pred(self):
        return self._ac_chunk * self._ac_dim
    
    def unflatten_ac_tensor(self, ac_tensor):
        out_shape = list(ac_tensor.shape[:-1]) + [self._ac_chunk, self._ac_dim]
        return ac_tensor.reshape(out_shape)
    
    def get_actions(self, inputs, zero_std=True):
        acs = self._sample(inputs, zero_std)
        return self.unflatten_ac_tensor(acs)
    
    def _sample(self, inputs, zero_std=True):
        dist = self(inputs, zero_std)
        return dist.sample()


class Deterministic(ActionDistribution):
    def __init__(self, in_dim, ac_dim, ac_chunk=1):
        super().__init__(ac_dim, ac_chunk)
        self._layer = nn.Linear(in_dim, self.num_ac_pred)
        
    def forward(self, inputs, zero_std=True):
        assert zero_std, "No std prediction in this network!"
        return self._layer(inputs)
    
    def _sample(self, inputs, zero_std=True):
        return self(inputs, zero_std)


class Gaussian(ActionDistribution):
    def __init__(self, in_dim, ac_dim, ac_chunk=1, min_std=1e-4, tanh_mean=False):
        super().__init__(ac_dim, ac_chunk)
        self._min_std, self._tanh_mean = min_std, tanh_mean        
        self._mean_net  = nn.Linear(in_dim, self.num_ac_pred)
        self._scale_net = nn.Linear(in_dim, self.num_ac_pred)

    def forward(self, in_repr, zero_std=False):
        B = in_repr.shape[0]
        mean = self._mean_net(in_repr).reshape(B, self.num_ac_pred)
        scale = self._scale_net(in_repr).reshape(B, self.num_ac_pred)

        # bound the action means and convert scale to std
        if self._tanh_mean:
            mean = torch.tanh(mean)
        std = torch.ones_like(scale) * self._min_std if zero_std else \
              F.softplus(scale) + self._min_std

        # create Normal action distributions
        return D.Normal(loc=mean, scale=std)


class GaussianSharedScale(ActionDistribution):
    def __init__(self, in_dim, ac_dim, ac_chunk=1, min_std=1e-4, tanh_mean=False,
                       log_std_init=0, std_fixed=False):
        super().__init__(ac_dim, ac_chunk)
        self._min_std, self._tanh_mean = min_std, tanh_mean
        self._mean_net = nn.Linear(in_dim, self.num_ac_pred)

        # create log_std vector and store as param
        log_std = torch.Tensor([log_std_init] * ac_dim)
        self.register_parameter('log_std', nn.Parameter(log_std, requires_grad=not std_fixed))

    def forward(self, in_repr, zero_std=False):
        B = in_repr.shape[0]
        mean = self._mean_net(in_repr).reshape(B, self.num_ac_pred)
        scale = self.log_std[None].repeat((B, self._ac_chunk))

        if self._tanh_mean:
            mean = torch.tanh(mean)
        std = torch.ones_like(scale) * self._min_std if zero_std else \
              torch.exp(scale) + self._min_std

        # create Normal action distributions
        return D.Normal(loc=mean, scale=std)


class GaussianMixture(ActionDistribution):
    def __init__(self, num_modes, in_dim, ac_dim, ac_chunk=1, min_std=1e-4, tanh_mean=False):
        super().__init__(ac_dim, ac_chunk)
        self._min_std, self._tanh_mean = min_std, tanh_mean
        self._num_modes = num_modes
        
        self._mean_net  = nn.Linear(in_dim, num_modes * self.num_ac_pred)
        self._scale_net = nn.Linear(in_dim, num_modes * self.num_ac_pred)
        self._logit_net = nn.Linear(in_dim, num_modes)

    def forward(self, in_repr, zero_std=False):
        B = in_repr.shape[0]
        mean = self._mean_net(in_repr).reshape(B, self._num_modes, self.num_ac_pred)
        scale = self._scale_net(in_repr).reshape(B, self._num_modes, self.num_ac_pred)
        logits = self._logit_net(in_repr).reshape((B, self._num_modes))

        # bound the action means and convert scale to std
        if self._tanh_mean:
            mean = torch.tanh(mean)
        std = torch.ones_like(scale) * self._min_std if zero_std else \
              F.softplus(scale) + self._min_std

        # create num_modes independent action distributions
        ac_dist = D.Normal(loc=mean, scale=std)
        ac_dist = D.Independent(ac_dist, 1)

        # parameterize the mixing distribution and the final GMM
        mix_dist = D.Categorical(logits=logits)
        gmm_dist = D.MixtureSameFamily(mixture_distribution=mix_dist,
                                       component_distribution=ac_dist)
        return gmm_dist
