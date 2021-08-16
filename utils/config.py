from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

import os


def _get_default_config():
  c = edict()

  # dataset
  c.data = edict()
  c.data.name = 'DefaultDataset'
  c.data.dir = './data'
  c.data.params = edict()

  # student model
  c.model = edict()
  c.model.params = edict()
  c.model.pretrain = edict()

  # train
  c.train = edict()
  c.train.dir = './result/out'
  c.train.batch_size = 64
  c.train.num_epochs = 2000
  c.train.num_grad_acc = None

  # evaluation
  c.eval = edict()
  c.eval.batch_size = 64

  # optimizer
  c.optimizer = edict()
  c.optimizer.name = 'adam'
  c.optimizer.params = edict()
  c.optimizer.gradient_clip = edict()

  # q_optimizer
  c.q_optimizer = edict()
  c.q_optimizer.name = 'adam'
  c.q_optimizer.params = edict()
  c.q_optimizer.gradient_clip = edict()

  # scheduler
  c.scheduler = edict()
  c.scheduler.name = 'none'
  c.scheduler.params = edict()

  # transforms
  c.transform = edict()
  c.transform.name = 'default_transform'
  c.transform.num_preprocessor = 4
  c.transform.params = edict()

  # losses
  c.loss = edict()
  c.loss.name = None
  c.loss.params = edict()

  return c


def _merge_config(src, dst):
  if not isinstance(src, edict):
    return

  for k, v in src.items():
    if isinstance(v, edict):
      _merge_config(src[k], dst[k])
    else:
      dst[k] = v


def load(config_path):
  with open(config_path, 'r') as fid:
    yaml_config = edict(yaml.load(fid))

  config = _get_default_config()
  _merge_config(yaml_config, config)

  set_model_weight_dirs(config)

  return config

def set_model_weight_dirs(config):

  if 'name' in config.model:
    model_dir = os.path.join(config.train.dir, config.model.name)
    config.train.model_dir = model_dir + config.train.model_dir
