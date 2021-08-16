from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.optim as optim


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001,
         amsgrad=False, **_):
  if isinstance(betas, str):
    betas = eval(betas)
  return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                    amsgrad=amsgrad)


def sgd(parameters, lr=0.1, momentum=0.9, weight_decay=0.0001, **_):
  return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_optimizer(config, parameters):
  f = globals().get(config.optimizer.name)
  return f(parameters, **config.optimizer.params)

def get_q_optimizer(config, parameters):
  f = globals().get(config.q_optimizer.name)
  return f(parameters, **config.q_optimizer.params)

def get_t_optimizer(config, parameters):
  f = globals().get(config.q_optimizer.name)
  return f(parameters, **config.t_optimizer.params)

def get_s_optimizer(config, parameters):
  f = globals().get(config.s_optimizer.name)
  return f(parameters, **config.s_optimizer.params)

def get_w_optimizer(config, parameters):
  f = globals().get(config.w_optimizer.name)
  return f(parameters, **config.w_optimizer.params)

def get_q_w_optimizer(config, parameters):
  f = globals().get(config.q_w_optimizer.name)
  return f(parameters, **config.q_w_optimizer.params)

def get_q_a_optimizer(config, parameters):
  f = globals().get(config.q_a_optimizer.name)
  return f(parameters, **config.q_a_optimizer.params)
