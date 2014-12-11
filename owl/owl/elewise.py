#!/usr/bin/env python

import libowl as _owl

mult = _owl.mult
exp = _owl.exp
ln = _owl.exp
sigm = _owl.sigm
relu = _owl.relu
tahn = _owl.tahn
sigm_back = _owl.sigm_back
#relu_back = _owl.relu_back
def relu_back(sens, acts):
    return _owl.relu_back(sens, acts, acts)
tahn_back = _owl.tahn_back
