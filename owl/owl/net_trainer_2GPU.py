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
    gpu = []
    gpu.append(owl.create_gpu_device(0))
    gpu.append(owl.create_gpu_device(1))
    owl.set_device(gpu[0])
    
    #prepare the net and solver
    builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
    owl_net = net.Net()
    builder.build_net(owl_net)
    builder.init_net_from_file(owl_net, '/home/tianjun/releaseversion/minerva/owl/apps/imagenet_googlenet/Googmodel/epoch0/')
    #builder.init_net_from_file(owl_net, '/home/tianjun/releaseversion/minerva/owl/apps/imagenet_googlenet/VGGmodel/epoch0/')
    #builder.init_net_from_file(owl_net, '/home/tianjun/releaseversion/minerva/owl/apps/imagenet_googlenet/Alexmodel/epoch0/')
    
    #set the accuracy layer
    acc_name = 'loss3/top-1'
    #acc_name = 'accuracy'
    last = time.time()
    
    count = 0

    wgrad = [[] for i in xrange(2)]
    bgrad = [[] for i in xrange(2)]
    num_samples = 0
    weights_id = get_weights_id(owl_net)

    for iteridx in xrange(owl_net.solver.max_iter):
        count = count + 1
        gpuid = count % 2

        owl.set_device(gpu[gpuid]) 
        
        owl_net.forward('TRAIN')
        owl_net.backward('TRAIN')
        num_samples += owl_net.units[builder.top_name_to_layer[acc_name][0]].batch_size

        #get the grad
        for widx in xrange(len(weights_id)):
            wgrad[gpuid].append(owl_net.units[weights_id[widx]].weightgrad)
            bgrad[gpuid].append(owl_net.units[weights_id[widx]].biasgrad)
        
        if count % 2 != 0:
            continue
       
        #merge grad
        for k in xrange(len(weights_id)):
            wgrad[0][k] += wgrad[1][k]
            bgrad[0][k] += bgrad[1][k]
            #update into owl_net
            owl_net.units[weights_id[k]].weight += (owl_net.base_lr * wgrad[0][k])
            owl_net.units[weights_id[k]].bias += (owl_net.base_lr * bgrad[0][k])

        #owl_net.units[builder.top_name_to_layer['prob']].to_numpy()
        accunit = owl_net.units[builder.top_name_to_layer[acc_name][0]]
        print "time: %s" % (time.time() - last)
        print accunit.acc
        last = time.time()
        
        wgrad = [[] for i in xrange(2)]
        bgrad = [[] for i in xrange(2)]
        num_samples = 0

