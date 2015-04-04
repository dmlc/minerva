#!/env/python

import math
import sys, argparse
import time
import numpy as np
import owl
import owl.net as net
from owl.net_helper import CaffeNetBuilder

class NetTrainer:
    def __init__(self, solver_file, snapshot, num_gpu = 1):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.num_gpu = num_gpu
        self.gpu = [owl.create_gpu_device(i) for i in range(num_gpu)]

    def build_net(self):
        self.owl_net = net.Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net, self.num_gpu)
        self.owl_net.init_layer_size('MULTI_VIEW')
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s):
        #multi-view test
        acc_num = 0
        test_num = 0
        loss_unit = s.owl_net.units[s.owl_net.name_to_uid['loss'][0]] 
        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            for i in range(10): 
                s.owl_net.forward('MULTI_VIEW')
                if i == 0:
                    softmax_val = loss_unit.ff_y
                    batch_size = softmax_val.shape[1]
                    softmax_label = loss_unit.y
                else:
                    softmax_val = softmax_val + loss_unit.ff_y
            
            test_num += batch_size
            predict = softmax_val.argmax(0)
            truth = softmax_label.argmax(0)
            correct = (predict - truth).count_zero()
            acc_num += correct
            print "Accuracy the %d mb: %f" % (testiteridx, correct)
            sys.stdout.flush()
        print "Testing Accuracy: %f" % (float(acc_num)/test_num)
   

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_file', help='caffe solver configure file')
    parser.add_argument('-n', '--num_gpu', help='number of gpus to use', action='store', type=int, default=1)
    parser.add_argument('--snapshot', help='the snapshot idx to start from', action='store', type=int)
    (args, remain) = parser.parse_known_args()
    solver_file = args.solver_file
    num_gpu = args.num_gpu
    snapshot = args.snapshot

    print ' === Using %d gpus, start from snapshot %d === ' % (num_gpu, snapshot)

    sys_args = [sys.argv[0]] + remain
    owl.initialize(sys_args)
    trainer = NetTrainer(solver_file, snapshot, num_gpu)
    trainer.build_net()
    trainer.run()
