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
    owl.initialize(sys.argv)
    gpu = [None] * 4
    gpu[0] = owl.create_gpu_device(0)
    gpu[1] = owl.create_gpu_device(1)
    gpu[2] = owl.create_gpu_device(2)
    gpu[3] = owl.create_gpu_device(3)
    
    #prepare the net and solver
    builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
    owl_net = net.Net()
    builder.build_net(owl_net)
    builder.init_net_from_file(owl_net, sys.argv[3])
    accunitname = sys.argv[4]
    last = time.time()

    wunits = get_weights_id(owl_net)
    print len(wunits)
    wgrad = [[] for i in xrange(4)] 
    bgrad = [[] for i in xrange(4)]
    
    for iteridx in range(owl_net.solver.max_iter):
        gpuidx = iteridx % 4
        owl.set_device(gpu[gpuidx])
        owl_net.forward('TRAIN')
        owl_net.backward('TRAIN')
            
        for wid in wunits:
            wgrad[gpuidx].append(owl_net.units[wid].weightgrad)
            bgrad[gpuidx].append(owl_net.units[wid].biasgrad)
        owl_net.get_units_by_name(accunitname)[0].ff_y.start_eval()
        
        if (iteridx+1) % 2 == 0:
            for i in range(len(wunits)):
                wid = wunits[i]
                wgrad[gpuidx][i] += wgrad[gpuidx-1][i]
                bgrad[gpuidx][i] += bgrad[gpuidx-1][i]

        if (iteridx+1) % 4 == 0:
            for i in range(len(wunits)):
                wid = wunits[i]
                wgrad[gpuidx][i] += wgrad[gpuidx-2][i]
                bgrad[gpuidx][i] += bgrad[gpuidx-2][i]
                owl_net.units[wid].weightgrad = wgrad[gpuidx][i]
                owl_net.units[wid].biasgrad = bgrad[gpuidx][i]

            wgrad = [[] for i in xrange(4)] 
            bgrad = [[] for i in xrange(4)]
            owl_net.weight_update()
            owl_net.get_units_by_name(accunitname)[0].ff_y.wait_for_eval()
            #print owl_net.get_units_by_name('accuracy')[0].acc
            print "Finished training 1 minibatch"
            print "time: %s" % (time.time() - last)
            last = time.time()

        '''
        #decide whether to test
        if (iteridx + 1) % owl_net.solver.test_interval == 0:
            acc_num = 0
            test_num = 0
            for testiteridx in range(owl_net.solver.test_iter):
                owl_net.forward('TEST')
                accunit = owl_net.get_units_by_name('loss3/top-1')[0]
                print "Accuracy this mb: %f" % (accunit.acc)
        '''
