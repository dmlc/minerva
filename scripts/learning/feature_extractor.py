#!/usr/bin/env python

import math
import sys, argparse
import time
import numpy as np
import owl
from owl.net.trainer import FeatureExtractor

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_file', help='caffe solver configure file')
    parser.add_argument('layer_name', help='layer_name')
    parser.add_argument('feature_path', help='feature_path')
    parser.add_argument('snapshot', help='the snapshot idx for test', type=int, default=0)
    parser.add_argument('gpu_idx', help='gpu to use', type=int, default=0)

    (args, remain) = parser.parse_known_args()
    solver_file = args.solver_file
    layer_name = args.layer_name
    feature_path = args.feature_path
    gpu_idx = args.gpu_idx
    snapshot = args.snapshot

    print ' === Extract feature of snapshot %d, using gpu #%d === ' % (snapshot, gpu_idx)

    sys_args = [sys.argv[0]] + remain

    extractor = FeatureExtractor(solver_file, snapshot, gpu_idx)
    extractor.build_net()
    extractor.run(layer_name, feature_path)
