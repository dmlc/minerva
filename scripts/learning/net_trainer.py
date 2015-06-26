#!/usr/bin/env python

import math
import sys, argparse
import time
import numpy as np
import owl
import owl.net as net
from owl.net.trainer import NetTrainer

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_file', help='caffe solver configure file')
    parser.add_argument('snapshot', help='the snapshot idx to start from', type=int, default=0)
    parser.add_argument('num_gpu', help='number of gpus to use', type=int, default=1)
    parser.add_argument('freq', help='frequency (number of minibatches) to call wait_for_all', type=int, default=1)

    (args, remain) = parser.parse_known_args()
    solver_file = args.solver_file
    num_gpu = args.num_gpu
    snapshot = args.snapshot
    freq = args.freq

    print ' === Training using %d gpus, start from snapshot %d === ' % (num_gpu, snapshot)

    sys_args = [sys.argv[0]] + remain
    trainer = NetTrainer(solver_file, snapshot, num_gpu, freq)
    trainer.build_net()
    trainer.run()
