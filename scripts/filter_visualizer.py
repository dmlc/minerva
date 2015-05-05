#!/usr/bin/env python

import math
import sys, argparse
import time
import numpy as np
import owl
from owl.net.trainer import FilterVisualizer

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_file', help='caffe solver configure file')
    parser.add_argument('snapshot', help='the snapshot idx to start from', action='store', type=int, default=0)
    parser.add_argument('layer_name', help='layer_name')
    parser.add_argument('result_path', help='result_path')
    parser.add_argument('-g', '--gpu_idx', help='gpu to use', action='store', type=int, default=0)

    (args, remain) = parser.parse_known_args()
    solver_file = args.solver_file
    layer_name = args.layer_name
    result_path = args.result_path
    gpu_idx = args.gpu_idx
    snapshot = args.snapshot

    print ' == Visualizing filter of snapshot %d, using gpu #%d === ' % (snapshot, gpu_idx)

    sys_args = [sys.argv[0]] + remain
    owl.initialize(sys_args)

    visualizer = FilterVisualizer(solver_file, snapshot, layer_name, result_path, gpu_idx)
    visualizer.build_net()
    visualizer.run()
