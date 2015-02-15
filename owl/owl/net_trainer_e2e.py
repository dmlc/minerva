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
    gpu = [None] * 2
    gpu[0] = owl.create_gpu_device(0)
    gpu[1] = owl.create_gpu_device(1)
    num_gpu = 2

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
    wgrad = []
    bgrad = []
    
    for iteridx in range(startsnapshot * owl_net.solver.snapshot * num_gpu, owl_net.solver.max_iter * num_gpu):
        #get the learning rate 
        if owl_net.solver.lr_policy == "poly":
            owl_net.current_lr = owl_net.base_lr * pow(1 - float(iteridx / num_gpu) / owl_net.solver.max_iter, owl_net.solver.power) 
        elif owl_net.solver.lr_policy == "step":
            owl_net.current_lr = owl_net.base_lr * pow(owl_net.solver.gamma, (iteridx / num_gpu) / owl_net.solver.step)

        owl.set_device(gpu[iteridx % 2])
        owl_net.forward('TRAIN')
        owl_net.backward('TRAIN')
        if iteridx % 2 == 0:
            for wid in wunits:
                wgrad.append(owl_net.units[wid].weightgrad)
                bgrad.append(owl_net.units[wid].biasgrad)
            owl_net.get_units_by_name(evallayername)[0].ff_y.start_eval()
        else:
            for i in range(len(wunits)):
                wid = wunits[i]
                owl_net.units[wid].weightgrad += wgrad[i]
                owl_net.units[wid].biasgrad += bgrad[i]
            wgrad = []
            bgrad = []
            owl_net.weight_update(2)
            owl_net.get_units_by_name(evallayername)[0].ff_y.wait_for_eval()
            print "Finished training %d minibatch" % (iteridx / 2)
            thistime = time.time() - last
            print "time: %s" % (thistime)
            last = time.time()
        
        if (iteridx + 1) % (owl_net.solver.display * num_gpu) == 0:
            lossunit = owl_net.get_units_by_name(evallayername)[0]
            print "Training Loss: %f" % (lossunit.getloss())
        
        #decide whether to test
        if (iteridx + 1) % (owl_net.solver.test_interval * num_gpu) == 0:
        #if True:
            acc_num = 0
            test_num = 0
            for testiteridx in range(owl_net.solver.test_iter[0]):
                owl_net.forward('TEST')
                accunit = owl_net.get_units_by_name(acclayername)[0]
                test_num += accunit.batch_size
                acc_num += (accunit.batch_size * accunit.acc)
                print "Accuracy the %d mb: %f" % (testiteridx, accunit.acc)
            print "Testing Accuracy: %f" % (float(acc_num)/test_num)
        
        if (iteridx + 1) % (owl_net.solver.snapshot * num_gpu) == 0:
            print "Save to snapshot %d, current lr %f" % ((iteridx + 1) / (owl_net.solver.snapshot * num_gpu) + startsnapshot, owl_net.current_lr)
            builder.save_net_to_file(owl_net, sys.argv[3], (iteridx + 1) / (owl_net.solver.snapshot * num_gpu) + startsnapshot)
            #print owl_net.current_lr
        sys.stdout.flush()

