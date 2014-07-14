#!/usr/bin/env python

from owl.random import randn
from owl import logical_dag

def main():
    n = 10
    k = 8
    x = randn(n, k)  # XXX: What does parts {1, 2} mean?
    y = randn(n, k)
    theta = randn(k, k)
    alpha = 0.5
    epoch = 2
    for i in xrange(epoch):
        error = x * theta - y
        theta = theta - alpha * x.trans() * error  # TODO: x.T
    with open('ldag.txt', 'w') as f:
        f.write(logical_dag() + '\n')

if __name__ == '__main__':
    main()
