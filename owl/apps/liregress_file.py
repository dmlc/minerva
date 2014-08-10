#!/usr/bin/env python

import sys

import owl
from owl import SimpleFileLoader
from owl import FileFormat


def main(args=sys.argv[1:]):
    owl.initialize(sys.argv)
    n = 10
    k = 8
    loader = SimpleFileLoader()
    x = owl.load_from_file((n, k), 'x.dat', loader, (1, 1))
    y = owl.load_from_file((n, k), 'y.dat', loader, (1, 1))
    theta = owl.load_from_file((k, k), 'theta.dat', loader, (1, 1))
    alpha = 0.5
    epoch = 2
    for i in xrange(epoch):
        error = x * theta - y
        theta = theta - alpha * x.trans() * error  # TODO: x.T
    format = FileFormat()
    format.binary = False
    theta.to_file('theta_trained.txt', format)
    format.binary = True
    theta.to_file('theta_trained.dat', format)
    print 'linear regression finished.'


if __name__ == '__main__':
    sys.exit(main())
