#!/usr/bin/env python

import sys
from libowl import *

def main():
    n = 10
    k = 8
    x = randn([n, k], 0.0, 0.1, [1, 1])
    y = randn([n, k], 0.0, 0.1, [1, 1])
    theta = randn([k, k], 0.0, 0.1, [1, 1])
    alpha = 0.5
    epoch = 2
    for i in xrange(epoch):
        error = x * theta - y
        theta = theta - alpha * x.trans() * error  # TODO: x.T
    theta.tofile('theta.txt');
    with open('ldag.txt', 'w') as f:
        f.write(logical_dag() + '\n')

if __name__ == '__main__':
    initialize(sys.argv);
    main()
