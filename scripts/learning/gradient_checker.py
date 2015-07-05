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
    parser.add_argument('checklayer', help='the layer to be checked')
    parser.add_argument('snapshot', help='the snapshot idx to start from', type=int, default=0)

    (args, remain) = parser.parse_known_args()
    solver_file = args.solver_file
    checklayer = args.checklayer
    snapshot = args.snapshot

    print ' ===  Gradient Check, start from snapshot %d === ' % (snapshot)

    sys_args = [sys.argv[0]] + remain
    trainer = NetTrainer(solver_file, snapshot, 1)
    trainer.build_net()
    trainer.gradient_checker(checklayer)
