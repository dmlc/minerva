#!/usr/bin/env python

import libowl as _owl

def randn(*args):
    # TODO: Check arguments
    scale = _owl.Scale(*args)
    parts = _owl.Scale(1, 2)
    array = _owl.random_randn(scale, 0.0, 1.0, parts)
    return array  # TODO: Wrap in a Python object
