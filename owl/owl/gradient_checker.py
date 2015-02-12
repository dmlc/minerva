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


def get_loss_layer(owl_net):
    weights_id = []
    for i in xrange(len(owl_net.units)):
        if isinstance(owl_net.units[i], net.SoftmaxUnit):
            weights_id.append(i)
    return weights_id

def check_bias(owl_net, checklayer):
    h = 1e-2
    threshold = 1e-4

    for iteridx in range(10):
        #disturb the weights
        oribias = checklayer.bias
        npbias = checklayer.bias.to_numpy()
        biasshape = np.shape(npbias)
        npbias = npbias.reshape(np.prod(biasshape[0:len(biasshape)]))
        print np.shape(npbias)
        position = np.random.randint(0, np.shape(npbias)[0]) 
        print position
        disturb = np.zeros(np.shape(npbias), dtype = np.float32)
        disturb[position] = h
        oriposval = npbias[position]
        npbias += disturb
        newposval = npbias[position]
        npbias = npbias.reshape(biasshape)
        checklayer.bias = owl.from_numpy(npbias)

        #get disturbed loss
        owl_net.forward('TRAIN')
        all_loss = 0
        for i in xrange(len(losslayer)):
            all_loss += owl_net.units[losslayer[i]].getloss()
        all_loss = all_loss / owl_net.batch_size #+ 0.5 * owl_net.base_weight_decay * newposval * newposval
        #get origin loss
        checklayer.bias = oribias
        owl_net.forward('TRAIN')
        ori_all_loss = 0
        for i in xrange(len(losslayer)):
            ori_all_loss += owl_net.units[losslayer[i]].getloss()
        ori_all_loss = ori_all_loss / owl_net.batch_size #+ 0.5 * owl_net.base_weight_decay * oriposval * oriposval
        owl_net.backward('TRAIN')
        #get analytic gradient
        npgrad = checklayer.biasgrad.to_numpy()
        npgrad = npgrad.reshape(np.prod(biasshape[0:len(biasshape)]))
        analy_grad = npgrad[position] / owl_net.batch_size
        #get numerical gradient
        
        print all_loss
        print ori_all_loss
        
        num_grad = (all_loss - ori_all_loss) / h

        diff = np.abs(analy_grad - num_grad)
        info = "analy: %f num: %f ratio: %f" % (analy_grad, num_grad, analy_grad / num_grad)
        print info


def check_weight(owl_net, checklayer):
    h = 1e-2
    threshold = 1e-4

    for iteridx in range(10):
        #disturb the weights
        oriweight = checklayer.weight
        npweight = checklayer.weight.to_numpy()
        weightshape = np.shape(npweight)
        npweight = npweight.reshape(np.prod(weightshape[0:len(weightshape)]))
        print np.shape(npweight)
        position = np.random.randint(0, np.shape(npweight)[0]) 
        print position
        disturb = np.zeros(np.shape(npweight), dtype = np.float32)
        disturb[position] = h
        oriposval = npweight[position]
        npweight += disturb
        newposval = npweight[position]
        npweight = npweight.reshape(weightshape)
        checklayer.weight = owl.from_numpy(npweight)

        #get disturbed loss
        owl_net.forward('TRAIN')
        all_loss = 0
        for i in xrange(len(losslayer)):
            all_loss += owl_net.units[losslayer[i]].getloss()
        all_loss = all_loss / owl_net.batch_size #+ 0.5 * owl_net.base_weight_decay * newposval * newposval
        #get origin loss
        checklayer.weight = oriweight
        owl_net.forward('TRAIN')
        ori_all_loss = 0
        for i in xrange(len(losslayer)):
            ori_all_loss += owl_net.units[losslayer[i]].getloss()
        ori_all_loss = ori_all_loss / owl_net.batch_size #+ 0.5 * owl_net.base_weight_decay * oriposval * oriposval
        owl_net.backward('TRAIN')
        #get analytic gradient
        npgrad = checklayer.weightgrad.to_numpy()
        npgrad = npgrad.reshape(np.prod(weightshape[0:len(weightshape)]))
        analy_grad = npgrad[position] / owl_net.batch_size
        #get numerical gradient
        
        print all_loss
        print ori_all_loss
        
        num_grad = (all_loss - ori_all_loss) / h

        diff = np.abs(analy_grad - num_grad)
        info = "analy: %f num: %f ratio: %f" % (analy_grad, num_grad, analy_grad / num_grad)
        print info

def check_weight_2gpu(owl_net, checklayer, gpu):
    h = 1e-2
    threshold = 1e-4
    wunits = get_weights_id(owl_net)
    wgrad = []
    bgrad = []

    for iteridx in range(10):
        #disturb the weights
        oriweight = checklayer.weight
        npweight = checklayer.weight.to_numpy()
        weightshape = np.shape(npweight)
        npweight = npweight.reshape(np.prod(weightshape[0:len(weightshape)]))
        print np.shape(npweight)
        position = np.random.randint(0, np.shape(npweight)[0]) 
        print position
        disturb = np.zeros(np.shape(npweight), dtype = np.float32)
        disturb[position] = h
        oriposval = npweight[position]
        npweight += disturb
        newposval = npweight[position]
        npweight = npweight.reshape(weightshape)
        checklayer.weight = owl.from_numpy(npweight)

        #get disturbed loss
        owl_net.forward('TRAIN')
        all_loss = 0
        for i in xrange(len(losslayer)):
            all_loss += owl_net.units[losslayer[i]].getloss()
        all_loss = all_loss / owl_net.batch_size #+ 0.5 * owl_net.base_weight_decay * newposval * newposval
        
        #get origin loss
        checklayer.weight = oriweight
        owl_net.forward('TRAIN')
        ori_all_loss = 0
        for i in xrange(len(losslayer)):
            ori_all_loss += owl_net.units[losslayer[i]].getloss()
        ori_all_loss = ori_all_loss / owl_net.batch_size #+ 0.5 * owl_net.base_weight_decay * oriposval * oriposval
        
        #analy_grad
        owl.set_device(gpu[0])
        owl_net.forward('TRAIN')
        owl_net.backward('TRAIN')
        for wid in wunits:
            wgrad.append(owl_net.units[wid].weightgrad)
            bgrad.append(owl_net.units[wid].biasgrad)
        owl.set_device(gpu[1])
        owl_net.forward('TRAIN')
        owl_net.backward('TRAIN')
        for i in range(len(wunits)):
            wid = wunits[i]
            owl_net.units[wid].weightgrad += wgrad[i]
            owl_net.units[wid].biasgrad += bgrad[i]
        wgrad = []
        bgrad = []
                
        #get analytic gradient
        npgrad = checklayer.weightgrad.to_numpy()
        npgrad = npgrad.reshape(np.prod(weightshape[0:len(weightshape)]))
        analy_grad = npgrad[position] / owl_net.batch_size / len(gpu)
        
        print all_loss
        print ori_all_loss
        num_grad = (all_loss - ori_all_loss) / h

        diff = np.abs(analy_grad - num_grad)
        info = "analy: %f num: %f ratio: %f" % (analy_grad, num_grad, analy_grad / num_grad)
        print info


if __name__ == "__main__":
    owl.initialize(sys.argv)
    gpu = []
    gpu.append(owl.create_gpu_device(0))
    gpu1 = owl.create_gpu_device(1)
    owl.set_device(gpu0)
    
    #prepare the net and solver
    builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
    owl_net = net.Net()
    builder.build_net(owl_net)
    builder.init_net_from_file(owl_net, sys.argv[3])
    accunitname = sys.argv[4]
    last = time.time()
    beg_time = last
   
    losslayer = get_loss_layer(owl_net)
    checklayer = owl_net.get_units_by_name(sys.argv[5])[0]

    check_weight(owl_net, checklayer)
    check_bias(owl_net, checklayer)
