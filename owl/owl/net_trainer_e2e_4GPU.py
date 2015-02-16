import math
import sys
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

if __name__ == "__main__":
    beg_time = time.time()
    owl.initialize(sys.argv)
    gpu = [None] * 4
    gpu[0] = owl.create_gpu_device(0)
    gpu[1] = owl.create_gpu_device(1)
    gpu[2] = owl.create_gpu_device(2)
    gpu[3] = owl.create_gpu_device(3)
    num_gpu = len(gpu)

    #prepare the net and solver
    builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
    owl_net = net.Net()
    builder.build_net(owl_net, num_gpu)
    startsnapshot = int(sys.argv[4])
    builder.init_net_from_file(owl_net, sys.argv[3], startsnapshot)
    evallayername = sys.argv[5]
    acclayername = sys.argv[6]
    last = time.time()

    wunits = get_weights_id(owl_net)
    wgrad = [[] for i in xrange(num_gpu)] 
    bgrad = [[] for i in xrange(num_gpu)]
    
  
    for iteridx in range(startsnapshot * owl_net.solver.snapshot, owl_net.solver.max_iter):
        #get the learning rate 
        if owl_net.solver.lr_policy == "poly":
            owl_net.current_lr = owl_net.base_lr * pow(1 - float(iteridx) / owl_net.solver.max_iter, owl_net.solver.power) 
        elif owl_net.solver.lr_policy == "step":
            owl_net.current_lr = owl_net.base_lr * pow(owl_net.solver.gamma, iteridx / owl_net.solver.step)

        # train on multi-gpu
        for gpuid in range(0, num_gpu):
            owl.set_device(gpuid)
            owl_net.forward('TRAIN')
            owl_net.backward('TRAIN')

            for wid in wunits:
                wgrad[gpuid].append(owl_net.units[wid].weightgrad)
                bgrad[gpuid].append(owl_net.units[wid].biasgrad)
            owl_net.get_units_by_name(evallayername)[0].ff_y.start_eval()

            if gpuid % 2 == 1:
                for i in range(len(wunits)):
                    wid = wunits[i]
                    wgrad[gpuid][i] += wgrad[gpuid-1][i]
                    bgrad[gpuid][i] += bgrad[gpuid-1][i]

            if gpuid == 3:
                for i in range(len(wunits)):
                    wid = wunits[i]
                    wgrad[gpuid][i] += wgrad[gpuid-2][i]
                    bgrad[gpuid][i] += bgrad[gpuid-2][i]
                    owl_net.units[wid].weightgrad = wgrad[gpuid][i]
                    owl_net.units[wid].biasgrad = bgrad[gpuid][i]
                wgrad = [[] for i in xrange(num_gpu)] 
                bgrad = [[] for i in xrange(num_gpu)]
                owl_net.weight_update() # XXX why not set num_gpu=4 here?
                owl_net.get_units_by_name(evallayername)[0].ff_y.wait_for_eval()

        print "Finished training %d minibatch" % iteridx
        print "time: %s" % (time.time() - last)
        last = time.time()       
        # decide whether to display loss
        if (iteridx + 1) % owl_net.solver.display == 0:
            lossunit = owl_net.get_units_by_name(evallayername)[0]
            print "Training Loss: %f" % (lossunit.getloss())
        
        # decide whether to test
        if (iteridx + 1) % owl_net.solver.test_interval == 0:
            acc_num = 0
            test_num = 0
            for testiteridx in range(owl_net.solver.test_iter[0]):
                owl_net.forward('TEST')
                accunit = owl_net.get_units_by_name(acclayername)[0]
                test_num += accunit.batch_size
                acc_num += (accunit.batch_size * accunit.acc)
                print "Accuracy the %d mb: %f" % (testiteridx, accunit.acc)
                sys.stdout.flush()
            print "Testing Accuracy: %f" % (float(acc_num)/test_num)
        
        # decide whether to save model
        if (iteridx + 1) % owl_net.solver.snapshot == 0:
            builder.save_net_to_file(owl_net, sys.argv[3], (iteridx + 1) / owl_net.solver.snapshot + startsnapshot)
            #print owl_net.current_lr
        sys.stdout.flush()

