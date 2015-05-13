#!/usr/bin/env python

import math
import sys, argparse
import time
import numpy as np
import owl
from owl.net.trainer import NetTester

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_file', help='caffe solver configure file')
    parser.add_argument('softmax_layer_name', help='softmax_layer_name')
    parser.add_argument('accuracy_layer_name', help='accuracy_layer_name')
    parser.add_argument('-s', '--snapshot', help='the snapshot idx for test', action='store', type=int, default=0)
    parser.add_argument('-g', '--gpu_idx', help='the gpu id to use', action='store', type=int, default=1)
    parser.add_argument('-m', '--multiview', help='whether to use multiview', action='store', type=int, default=False)

    (args, remain) = parser.parse_known_args()
    solver_file = args.solver_file
    softmax_layer_name = args.softmax_layer_name
    accuracy_layer_name = args.accuracy_layer_name
    snapshot = args.snapshot
    gpu_idx = args.gpu_idx
    multiview = bool(args.multiview)

    print ' === "Test for snapshot %d, using gpu #%d=== ' % (snapshot, gpu_idx)

    sys_args = [sys.argv[0]] + remain
    owl.initialize(sys_args)
    tester = NetTester(solver_file, softmax_layer_name, accuracy_layer_name, snapshot, gpu_idx)
    tester.build_net()
    tester.run(multiview)
