import math
import sys
import time
import numpy as np
import owl
import owl.net as net
from net_helper import CaffeNetBuilder

if __name__ == "__main__":
    owl.initialize(sys.argv)
    gpu0 = owl.create_gpu_device(0)
    owl.set_device(gpu0)
   
    builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
    owl_net = net.Net()
    builder.build_net(owl_net)
    builder.init_net_from_file(owl_net, '/home/tianjun/releaseversion/minerva/owl/apps/imagenet_googlenet/Googmodel/epoch0/')
    #ff, bp, updat
    owl_net.forward()
    owl_net.backward()
    owl_net.weight_update()
    #check acc
    acc_name = 'loss3/top-1'
    
    print "Accuracy: %f" % (owl_net.units[builder.top_name_to_layer[acc_name][0]].acc)




