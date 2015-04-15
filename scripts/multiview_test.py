#!/usr/bin/env python

import math
import sys, argparse
import time
import numpy as np
import owl
from owl.net.trainer import MultiviewTester

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_file', help='caffe solver configure file')
    parser.add_argument('snapshot', help='the snapshot idx for test', action='store', type=int, default=0)
    parser.add_argument('-g', '--gpu_idx', help='the gpu id to use', action='store', type=int, default=1)

    (args, remain) = parser.parse_known_args()
    solver_file = args.solver_file
    snapshot = args.snapshot
    gpu_idx = args.gpu_idx

    print ' === Multiview test for snapshot %d, using gpu #%d=== ' % (snapshot, gpu_idx)

    sys_args = [sys.argv[0]] + remain
    owl.initialize(sys_args)

    tester = MultiviewTester(solver_file, snapshot, gpu_idx)
    tester.build_net()
    tester.run()
