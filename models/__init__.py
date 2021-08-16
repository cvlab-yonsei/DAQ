#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

## resnet20 ##
from .resnet20_DAQ import *
from .resnet20_DAQ_w import *

def get_model(config):
    print('model name:', config.model.name)
    f = globals().get(config.model.name)
    if config.model.params is None:
        return f()
    else:
        return f(**config.model.params)

if __name__ == '__main__':
    print('main')
    model = get_resnet34()
