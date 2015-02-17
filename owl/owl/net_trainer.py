import math
import sys, getopt
import time
import numpy as np
import owl
import owl.net as net
from net_helper import CaffeNetBuilder

def get_weights_id(owl_net):
    weights_id = []
    for i in xrange(len(owl_net.units)):
        if isinstance(owl_net.units[i], net.WeightedComputeUnit):
            weights_id.append(i)
    return weights_id

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
        gpu = [None] * s.num_gpu
        for i in range(0, s.num_gpu):
            gpu[i] = owl.create_gpu_device(i)
        last = time.time()
        wunits = s.owl_net.get_weighted_unit_ids()
        wgrad = []
        bgrad = []
        
        for iteridx in range(s.snapshot * s.owl_net.solver.snapshot, s.owl_net.solver.max_iter):
            # get the learning rate 
            if s.owl_net.solver.lr_policy == "poly":
                s.owl_net.current_lr = s.owl_net.base_lr * pow(1 - float(iteridx) / s.owl_net.solver.max_iter, s.owl_net.solver.power) 
            elif s.owl_net.solver.lr_policy == "step":
                s.owl_net.current_lr = s.owl_net.base_lr * pow(s.owl_net.solver.gamma, iteridx / s.owl_net.solver.step)

            # train on multi-gpu
            for gpuid in range(0, s.num_gpu):
                owl.set_device(gpu[gpuid])
                s.owl_net.forward('TRAIN')
                s.owl_net.backward('TRAIN')
                if gpuid == 0:
                    for wid in wunits:
                        wgrad.append(s.owl_net.units[wid].weightgrad)
                        bgrad.append(s.owl_net.units[wid].biasgrad)
                    s.owl_net.start_eval_loss()
                else:
                    for i in range(len(wunits)):
                        wid = wunits[i]
                        s.owl_net.units[wid].weightgrad += wgrad[i]
                        s.owl_net.units[wid].biasgrad += bgrad[i]
                    s.owl_net.weight_update(num_gpu = 2)
                    s.owl_net.wait_for_eval_loss()
            wgrad = []
            bgrad = []

            print "Finished training %d minibatch" % (iteridx)
            thistime = time.time() - last
            print "time: %s" % (thistime)
            last = time.time()
            # decide whether to display loss
            if (iteridx + 1) % (s.owl_net.solver.display) == 0:
                lossunits = s.owl_net.get_loss_units()
                for lu in len(lossunit):
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
                print "Save to snapshot %d, current lr %f" % ((iteridx + 1) / (s.owl_net.solver.snapshot) + s.snapshot, s.owl_net.current_lr)
                builder.save_net_to_file(s.owl_net, s.snapshot_dir, (iteridx + 1) / (s.owl_net.solver.snapshot) + s.snapshot)
                #print s.owl_net.current_lr
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
    try:
        opts, args = getopt.getopt(sys.argv, 'hn:', ["help", "snapshot=", "snapshot-dir=", "num-gpu="])
    except getopt.GetoptError:
        print_help_and_exit()
    net_file = sys.argv[1]
    solver_file = sys.argv[2]
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help_and_exit()
        elif opt in ("-n", "--num-gpu"):
            num_gpu = int(arg)
        elif opt == "--snapshot":
            snapshot = arg
        elif opt == "--snapshot-dir":
            snapshot_dir = arg
        
    owl.initialize(sys.argv)
    trainer = NetTrainer(net_file, solver_file, snapshot, snapshot_dir, num_gpu)
    trainer.build_net()
    trainer.run()
