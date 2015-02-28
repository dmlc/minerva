#!/env/python

import math
import sys, getopt
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

    def build_net(self):
        self.owl_net = net.Net()
        builder = CaffeNetBuilder(self.net_file, self.solver_file)
        builder.build_net(self.owl_net, self.num_gpu)
        builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s):
        gpu = [owl.create_gpu_device(i) for i in range(s.num_gpu)]
        wgrad = [[] for i in range(s.num_gpu)]
        bgrad = [[] for i in range(s.num_gpu)]
        for i in range(0, s.num_gpu):
            gpu.append(owl.create_gpu_device(i))
            wgrad.append([])
            bgrad.append([])
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
                owl.set_device(gpu[gpuid])
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
                owl.set_device(gpu[upd_gpu])
                for gid in range(s.num_gpu):
                    if gid == upd_gpu:
                        continue
                    wgrad[upd_gpu][i] += wgrad[gid][i]
                    bgrad[upd_gpu][i] += bgrad[gid][i]
                s.owl_net.units[wid].weightgrad = wgrad[upd_gpu][i]
                s.owl_net.units[wid].biasgrad = bgrad[upd_gpu][i]
                s.owl_net.update(wid, s.num_gpu)
            #s.owl_net.weight_update(num_gpu = s.num_gpu)
            s.owl_net.wait_for_eval_loss()
            #s.owl_net.units[wunits[0]].weight.wait_for_eval()
            wgrad = [[] for i in range(s.num_gpu)] # reset gradients
            bgrad = [[] for i in range(s.num_gpu)]

            thistime = time.time() - last
            print "Finished training %d minibatch (time: %s)" % (iteridx, thistime)
            last = time.time()
            # decide whether to display loss
            if (iteridx + 1) % (s.owl_net.solver.display) == 0:
                lossunits = s.owl_net.get_loss_units()
                for lu in lossunits:
                    print "Training Loss %s: %f" % (lu.name, lu.getloss())
            
            # decide whether to test
            if (iteridx + 1) % (s.owl_net.solver.test_interval) == 0:
                acc_num = 0
                test_num = 0
                for testiteridx in range(s.owl_net.solver.test_iter[0]):
                    s.owl_net.forward('TEST')
                    accunit = s.owl_net.get_accuracy_units()[0]
                    test_num += accunit.batch_size
                    acc_num += (accunit.batch_size * accunit.acc)
                    print "Accuracy the %d mb: %f" % (testiteridx, accunit.acc)
                print "Testing Accuracy: %f" % (float(acc_num)/test_num)
            
            # decide whether to save model
            if (iteridx + 1) % (s.owl_net.solver.snapshot) == 0:
                print "Save to snapshot %d, current lr %f" % ((iteridx + 1) / (s.owl_net.solver.snapshot), s.owl_net.current_lr)
                builder.save_net_to_file(s.owl_net, s.snapshot_dir, (iteridx + 1) / (s.owl_net.solver.snapshot))
            sys.stdout.flush()

def print_help_and_exit():
    print """
    Usage: net_trainer.py <net_file> <solver_file> [options]\n
    Options:\n
            -h,--help        print this help
            --snapshot       start training from given snapshot
            --snapshot-dir   the root directory of snapshot
            -n,--num-gpu     number of gpus to use
    """
    sys.exit(2)



if __name__ == "__main__":
    # parse command line arguments
    if len(sys.argv) < 3:
        print_help_and_exit()
    net_file = sys.argv[1]
    solver_file = sys.argv[2]
        
    try:
        opts, args = getopt.getopt(sys.argv[3:], 'hn:', ["help", "snapshot=", "snapshot-dir=", "num-gpu="])
    except getopt.GetoptError:
        print_help_and_exit()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help_and_exit()
        elif opt in ("-n", "--num-gpu"):
            num_gpu = int(arg)
        elif opt == "--snapshot":
            snapshot = int(arg)
        elif opt == "--snapshot-dir":
            snapshot_dir = arg

    print ' === Using %d gpus, start from snapshot %d === ' % (num_gpu, snapshot)

    owl.initialize(sys.argv)
    trainer = NetTrainer(net_file, solver_file, snapshot, snapshot_dir, num_gpu)
    trainer.build_net()
    trainer.run()
