#!/env/python

import math
import sys, argparse
import time
import numpy as np
import owl
import owl.net as net
from owl.net_helper import CaffeNetBuilder

class NetTrainer:
    def __init__(self, net_file, solver_file, snapshot, snapshot_dir, num_gpu = 1):
        self.net_file = net_file
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.snapshot_dir = snapshot_dir
        self.num_gpu = num_gpu
        self.gpu = [owl.create_gpu_device(i) for i in range(num_gpu)]

    def build_net(self):
        self.owl_net = net.Net()
        self.builder = CaffeNetBuilder(self.net_file, self.solver_file)
        self.builder.build_net(self.owl_net, self.num_gpu)
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s):
        wgrad = [[] for i in range(s.num_gpu)]
        bgrad = [[] for i in range(s.num_gpu)]
        last = time.time()
        wunits = s.owl_net.get_weighted_unit_ids()
        last_start = time.time()
        
        for iteridx in range(s.snapshot * s.owl_net.solver.snapshot, s.owl_net.solver.max_iter):
            # get the learning rate 
            if s.owl_net.solver.lr_policy == "poly":
                s.owl_net.current_lr = s.owl_net.base_lr * pow(1 - float(iteridx) / s.owl_net.solver.max_iter, s.owl_net.solver.power) 
            elif s.owl_net.solver.lr_policy == "step":
                s.owl_net.current_lr = s.owl_net.base_lr * pow(s.owl_net.solver.gamma, iteridx / s.owl_net.solver.stepsize)
            # train on multi-gpu
            for gpuid in range(s.num_gpu):
                owl.set_device(s.gpu[gpuid])
                s.owl_net.forward('TRAIN')
                s.owl_net.backward('TRAIN')
                for wid in wunits:
                    wgrad[gpuid].append(s.owl_net.units[wid].weightgrad)
                    bgrad[gpuid].append(s.owl_net.units[wid].biasgrad)
                s.owl_net.start_eval_loss()

            # weight update
            for i in range(len(wunits)): 
                wid = wunits[i]
                upd_gpu = i * num_gpu / len(wunits)
                owl.set_device(s.gpu[upd_gpu])
                for gid in range(s.num_gpu):
                    if gid == upd_gpu:
                        continue
                    wgrad[upd_gpu][i] += wgrad[gid][i]
                    bgrad[upd_gpu][i] += bgrad[gid][i]
                s.owl_net.units[wid].weightgrad = wgrad[upd_gpu][i]
                s.owl_net.units[wid].biasgrad = bgrad[upd_gpu][i]
                s.owl_net.update(wid)
            #s.owl_net.weight_update(num_gpu = s.num_gpu)
            if iteridx % 2 == 0:
                s.owl_net.wait_for_eval_loss()
                thistime = time.time() - last
                print "Finished training %d minibatch (time: %s)" % (iteridx, thistime)
                last = time.time()
            else:
                s.owl_net.start_eval_loss()

            #s.owl_net.units[wunits[0]].weight.wait_for_eval()
            wgrad = [[] for i in range(s.num_gpu)] # reset gradients
            bgrad = [[] for i in range(s.num_gpu)]

            # decide whether to display loss
            if (iteridx + 1) % (s.owl_net.solver.display) == 0:
                lossunits = s.owl_net.get_loss_units()
                for lu in lossunits:
                    print "Training Loss %s: %f" % (lu.name, lu.getloss())
            
            # decide whether to test
            #if True:
            if (iteridx + 1) % (s.owl_net.solver.test_interval) == 0:
                acc_num = 0
                test_num = 0
                for testiteridx in range(s.owl_net.solver.test_iter[0]):
                    s.owl_net.forward('TEST')
                    all_accunits = s.owl_net.get_accuracy_units()
                    accunit = all_accunits[len(all_accunits)-1]
                    #accunit = all_accunits[0]
                    print accunit.name
                    test_num += accunit.batch_size
                    acc_num += (accunit.batch_size * accunit.acc)
                    print "Accuracy the %d mb: %f" % (testiteridx, accunit.acc)
                    sys.stdout.flush()
                print "Testing Accuracy: %f" % (float(acc_num)/test_num)
            
            # decide whether to save model
            if (iteridx + 1) % (s.owl_net.solver.snapshot) == 0:
                print "Save to snapshot %d, current lr %f" % ((iteridx + 1) / (s.owl_net.solver.snapshot), s.owl_net.current_lr)
                s.builder.save_net_to_file(s.owl_net, s.snapshot_dir, (iteridx + 1) / (s.owl_net.solver.snapshot))
            sys.stdout.flush()

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('net_file', help='caffe network configure file')
    parser.add_argument('solver_file', help='caffe solver configure file')
    parser.add_argument('-n', '--num_gpu', help='number of gpus to use', action='store', type=int, default=1)
    parser.add_argument('--snapshot', help='the snapshot idx to start from', action='store', type=int)
    parser.add_argument('--snapshot_dir', help='the root directory of snapshot', action='store', type=str)
    (args, remain) = parser.parse_known_args()
    net_file = args.net_file
    solver_file = args.solver_file
    num_gpu = args.num_gpu
    snapshot = args.snapshot
    snapshot_dir = args.snapshot_dir
    
    print ' === Using %d gpus, start from snapshot %d === ' % (num_gpu, snapshot)

    sys_args = [sys.argv[0]] + remain
    owl.initialize(sys_args)
    trainer = NetTrainer(net_file, solver_file, snapshot, snapshot_dir, num_gpu)
    trainer.build_net()
    trainer.run()
