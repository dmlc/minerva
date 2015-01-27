import sys
import math
import time
import numpy as np
from caffe_data_pb2 import NetParameter
from caffe_data_pb2 import LayerParameter
from google.protobuf import text_format
import owl
import owl.conv as conv
import owl.elewise as ele
import owl.net as net
from imageio import ImageNetDataProvider

class CaffeModelConfig:
    def __init__(self, configfile):
        file = open(configfile, "r")
        self.netconfig = NetParameter()        
        text_format.Merge(str(file.read()), self.netconfig)
        self.netconfig

class MinervaModel:
    def __init__(self, netconfig, weight_dir):
        self.layers = dict()
        self.connections = dict()
        
        self.inputlayer = []
        self.outputlayer = []

        layerparam = LayerParameter()
  
        #input can be defined in layers or out of layers
        if len(netconfig.input_dim) == 4:
            self.input_size = netconfig.input_dim[3]
            self.input_channel = netconfig.input_dim[1]
            #create input layer
            name = 'data'
            self.layers[name] = net.Layer(name, net.LinearNeuron(), 0)
            if name == 'data':
                self.layers[name].dim = [self.input_size, self.input_size, self.input_channel]
            self.inputlayer.append(name)
            

        for i in range(len(netconfig.layers)):
            #data is a input layer
            if netconfig.layers[i].type == layerparam.LayerType.Value('DATA'):
                self.input_size = netconfig.layers[i].transform_param.crop_size
                self.input_channel = 3
                for topidx in range(len(netconfig.layers[i].top)):
                    if netconfig.layers[i].top[topidx] not in self.inputlayer: 
                        name = netconfig.layers[i].top[topidx]
                        self.layers[name] = net.Layer(name, net.LinearNeuron(), 0)
                        if name == 'data':
                            self.layers[name].dim = [self.input_size, self.input_size, self.input_channel]
                        self.inputlayer.append(name)
            
            #if top[0] != bottom[0], it should be a connection, otherwise, we just consider it as a non-linear operations
            if len(netconfig.layers[i].top) >= 1 and  len(netconfig.layers[i].bottom) >= 1:
                #create connection
                if netconfig.layers[i].top[0] != netconfig.layers[i].bottom[0]:
                    print "Connection %d" % (i)
                    name = netconfig.layers[i].name
                    print netconfig.layers[i].name
                    this_dim = []

                    #decide connection type
                    if netconfig.layers[i].type == layerparam.LayerType.Value('CONVOLUTION'):
                        #get input information
                        assert len(netconfig.layers[i].top) == 1 and len(netconfig.layers[i].bottom) == 1
                        bot_name = netconfig.layers[i].bottom[0]
                        assert bot_name in self.layers
                        bot_dim = self.layers[bot_name].dim
                        #get convinfo
                        kernel_size = netconfig.layers[i].convolution_param.kernel_size
                        num_output = netconfig.layers[i].convolution_param.num_output
                        wshape = [kernel_size, kernel_size, bot_dim[2], num_output] 
                        bshape = [num_output]
                        #default pad = 0 stride = 1
                        pad = netconfig.layers[i].convolution_param.pad
                        stride = netconfig.layers[i].convolution_param.stride
                       
                        '''
                        #next scale
                        if pad > 0:
                            for right_pad in range(pad, 0, -1):
                                if (bot_dim[0] + pad + right_pad - kernel_size) % stride == 0:
                                    this_scale = (bot_dim[0] + pad + right_pad - kernel_size) / stride + 1
                                    break
                        else:
                            for right_pad in range(stride-1, -1, -1):
                                if (bot_dim[0] + right_pad - kernel_size) % stride == 0:
                                    this_scale = (bot_dim[0] + right_pad - kernel_size) / stride + 1
                                    break
                        '''

                        this_scale = (bot_dim[0] + pad * 2 - kernel_size) / stride + 1
                        this_dim = [this_scale, this_scale, num_output]
                        
                        #initer
                        #winiter = netconfig.layers[i].convolution_param.weight_filler.type
                        #biniter = netconfig.layers[i].convolution_param.bias_filler.type
                        #TODO: hack
                        layername = netconfig.layers[i].name
                        layername = layername.replace("/","_")
                        winiter = '%s/epoch0/%s_weights.dat' % (weight_dir, layername)
                        biniter = '%s/epoch0/%s_bias.dat' % (weight_dir, layername)

                        #blob learning rate and wd
                        self.blobs_lr = netconfig.layers[i].blobs_lr
                        self.blobs_wd = netconfig.layers[i].weight_decay
                        #create the cinnetion
                        self.connections[name] = net.ConvConnection(name, wshape, bshape, winiter, biniter, pad, stride) 
                    elif  netconfig.layers[i].type == layerparam.LayerType.Value('POOLING'):
                        assert len(netconfig.layers[i].top) == 1 and len(netconfig.layers[i].bottom) == 1
                        bot_name = netconfig.layers[i].bottom[0]
                        assert bot_name in self.layers
                        bot_dim = self.layers[bot_name].dim
                        #get poolinfo
                        kernel_size = netconfig.layers[i].pooling_param.kernel_size
                        pad = netconfig.layers[i].pooling_param.pad
                        stride = netconfig.layers[i].pooling_param.stride
                        pooltype = netconfig.layers[i].pooling_param.pool
                       
                        this_scale = int(np.ceil((float(bot_dim[0] + pad * 2) - kernel_size) / stride)) + 1
                        if (this_scale - 1) * stride >= bot_dim[0] + pad:
                            this_scale -= 1

                        this_dim = [this_scale, this_scale, bot_dim[2]]
                        if netconfig.layers[i].pooling_param.pool == 0:
                            op = conv.pool_op.max
                        elif netconfig.layers[i].pooling_param.pool == 1:
                            op = conv.pool_op.avg
                        else:
                            assert false
                        self.connections[name] = net.PoolingConnection(name, kernel_size, stride, pad, op)
                    elif  netconfig.layers[i].type == layerparam.LayerType.Value('INNER_PRODUCT'):
                        assert len(netconfig.layers[i].top) == 1 and len(netconfig.layers[i].bottom) == 1
                        bot_name = netconfig.layers[i].bottom[0]
                        assert bot_name in self.layers
                        bot_dim = self.layers[bot_name].dim
                        bot_dim_flat = 0
                        if len(bot_dim) > 1:
                            bot_dim_flat = np.prod(bot_dim)
                        else:
                            bot_dim_flat = bot_dim[0]
                        num_output = netconfig.layers[i].inner_product_param.num_output
                        this_dim = [netconfig.layers[i].inner_product_param.num_output]
                        #initer
                        #winiter = netconfig.layers[i].inner_product_param.weight_filler.type
                        #biniter = netconfig.layers[i].inner_product_param.bias_filler.type
                        #TODO: hack
                        layername = netconfig.layers[i].name
                        layername = layername.replace("/","_")
                        winiter = '%s/epoch0/%s_weights.dat' % (weight_dir, layername)
                        biniter = '%s/epoch0/%s_bias.dat' % (weight_dir, layername)
                        
                        #blob learning rate and wd
                        self.blobs_lr = netconfig.layers[i].blobs_lr
                        self.blobs_wd = netconfig.layers[i].weight_decay
                        #weight shape
                        wshape = [num_output, bot_dim_flat]
                        bshape = [num_output, 1]
                        self.connections[netconfig.layers[i].name] = net.FullyConnection(netconfig.layers[i].name, wshape, bshape, winiter, biniter)
                    elif  netconfig.layers[i].type == layerparam.LayerType.Value('LRN'):
                        assert len(netconfig.layers[i].top) == 1 and len(netconfig.layers[i].bottom) == 1
                        bot_name = netconfig.layers[i].bottom[0]
                        assert bot_name in self.layers
                        this_dim = self.layers[bot_name].dim
                        local_size = netconfig.layers[i].lrn_param.local_size
                        alpha = netconfig.layers[i].lrn_param.alpha
                        beta = netconfig.layers[i].lrn_param.beta
                        self.connections[netconfig.layers[i].name] = net.LRNConnection(netconfig.layers[i].name, local_size, alpha, beta)
                    elif netconfig.layers[i].type == layerparam.LayerType.Value('SOFTMAX_LOSS'):
                        assert len(netconfig.layers[i].top) == 1 and len(netconfig.layers[i].bottom) == 2
                        bot_name = netconfig.layers[i].bottom[0]
                        this_dim = self.layers[bot_name].dim
                        self.connections[netconfig.layers[i].name] = net.SoftMaxConnection(netconfig.layers[i].name)
                    elif netconfig.layers[i].type == layerparam.LayerType.Value('CONCAT'):
                        assert len(netconfig.layers[i].top) == 1 and len(netconfig.layers[i].bottom) > 1
                        concat_dim_caffe = netconfig.layers[i].concat_param.concat_dim
                        concat_dim = concat_dim_caffe
                        if concat_dim_caffe == 1:
                            assert len(self.layers[netconfig.layers[i].bottom[0]].dim) == 3
                            concat_dim = 2
                        else:
                            concat_dim = len(self.layers[netconfig.layers[i].bottom[0]])
                        #Get New Dim
                        this_dim = [self.layers[netconfig.layers[i].bottom[0]].dim[0], self.layers[netconfig.layers[i].bottom[1]].dim[1],self.layers[netconfig.layers[i].bottom[0]].dim[2]]
                        this_dim[concat_dim] = 0
                        for botidx in range(len(netconfig.layers[i].bottom)):
                            this_dim[concat_dim] += self.layers[netconfig.layers[i].bottom[botidx]].dim[concat_dim]
                        self.connections[netconfig.layers[i].name] = net.ConcatConnection(netconfig.layers[i].name, concat_dim)
                    else:
                        print 'Not Implemented Connection'
                        print netconfig.layers[i].name
                        continue
 
                    #create layeronents and it's name
                    name = netconfig.layers[i].top[0]
                    self.layers[name] = net.Layer(name, net.LinearNeuron(), 0)
                    self.layers[name].dim = this_dim
                    self.layers[netconfig.layers[i].top[0]].bp_conns.append(self.connections[netconfig.layers[i].name])
                    self.connections[netconfig.layers[i].name].top.append(self.layers[name])
                   
                    print self.layers[name].dim
                    
                    #output connection
                    for botidx in range(len(netconfig.layers[i].bottom)):
                        assert netconfig.layers[i].bottom[botidx] in self.layers
                        self.layers[netconfig.layers[i].bottom[botidx]].ff_conns.append(self.connections[netconfig.layers[i].name])
                        self.connections[netconfig.layers[i].name].bottom.append(self.layers[netconfig.layers[i].bottom[botidx]])
                    
                #other operations just seen as the non-linear operation
                elif len(netconfig.layers[i].top) == 1 and len(netconfig.layers[i].bottom) == 1 and netconfig.layers[i].top[0] == netconfig.layers[i].bottom[0]:
                    print "Neuron %d" % (i)

                    assert netconfig.layers[i].bottom[0] in self.layers
                    if netconfig.layers[i].type == layerparam.LayerType.Value('RELU'):
                        self.layers[netconfig.layers[i].bottom[0]].neuron = net.ReluNeuron()
                    elif netconfig.layers[i].type == layerparam.LayerType.Value('DROPOUT'):
                        self.layers[netconfig.layers[i].bottom[0]].dropout = netconfig.layers[i].dropout_param.dropout_ratio
                    else:
                        print 'Not Implemented Neuron'
                        print netconfig.layers[i].name
                        assert False
                else:
                    assert netconfig.layers[i].type == layerparam.LayerType.Value('DATA')

    def ff(self, data, labels):
        print "======================="
        print "Network Feed Forward"
        #init status
        connstatus = dict()
        layerstatus = dict()
        jobqueue = []

        for conn in self.connections:
            connstatus[conn] = False
        for layer in self.layers:
            layerstatus[layer] = False
        
        #mark the inlayer ready
        for inlayer in self.inputlayer:
            print inlayer
            if inlayer == 'data':
                self.layers[inlayer].pre_act = data
            elif inlayer == 'label':
                self.layers[inlayer].pre_act = labels
            else:
                assert False
            layerstatus[inlayer] = True
        
        #mark the ready connection
        for inlayer in self.inputlayer:
            #check whether the inlayer related connection is ready
            for inlayer_conn in self.layers[inlayer].ff_conns:
                connstatus[inlayer_conn.name] = True
                for conn_bottom in inlayer_conn.bottom:
                    if layerstatus[conn_bottom.name] == False:
                        connstatus[inlayer_conn.name] = False
                        break
                if connstatus[inlayer_conn.name] == True:
                    jobqueue.insert(0, inlayer_conn)

        while(len(jobqueue)!=0):
            curjob = jobqueue.pop()
            #first connection ff, the output is the top's pre-nonlinear
            print curjob.name
            assert len(curjob.top) == 1
            curjob.top[0].pre_act = curjob.ff()
            #activate trough non-linear function and dropout
            curjob.top[0].ff()
 
            '''
            if curjob.name == "fc6":
                numpyarr = curjob.top[0].get_act().to_numpy()
                res = np.reshape(numpyarr, np.prod(np.shape(numpyarr))).tolist()[112*0:112*1+10]
                for i in range(len(res)):
                    print '%d:%f' % (i, res[i])
                print np.shape(numpyarr)
                exit(0)
            '''

            #trigger the followers
            for finishedlayer in self.connections[curjob.name].top:
                layerstatus[finishedlayer.name] = True
            for finishedlayer in self.connections[curjob.name].top:
                for next_conn in finishedlayer.ff_conns:
                    if connstatus[next_conn.name] == True:
                        continue
                    connstatus[next_conn.name] = True
                    for next_conn_bottom in next_conn.bottom:
                        if layerstatus[next_conn_bottom.name] == False:
                            connstatus[next_conn.name] = False
                            break
                    if connstatus[next_conn.name] == True:
                        jobqueue.insert(0, next_conn)
    
    def bp(self, data, labels):
        print "======================="
        print "Network Backward"
        #init status
        connstatus = dict()
        layerstatus = dict()
        jobqueue = []

        for conn in self.connections:
            connstatus[conn] = False
        for layer in self.layers:
            layerstatus[layer] = False
       
        outputlayer = []
        for nnlayer in self.layers:
            if len(self.layers[nnlayer].ff_conns) == 0:
                outputlayer.append(self.layers[nnlayer])

        #mark the inlayer ready
        for outlayer in outputlayer:
            #print outlayer.name
            layerstatus[outlayer.name] = True
        
        #mark the ready connection
        for outlayer in outputlayer:
            #check whether the inlayer related connection is ready
            for outlayer_conn in self.layers[outlayer.name].bp_conns:
                connstatus[outlayer_conn.name] = True
                for conn_top in outlayer_conn.top:
                    if layerstatus[conn_top.name] == False:
                        connstatus[outlayer_conn.name] = False
                        break
                if connstatus[outlayer_conn.name] == True:
                    jobqueue.insert(0, outlayer_conn)

        while(len(jobqueue)!=0):
            curjob = jobqueue.pop()
            #first connection ff, the output is the top's pre-nonlinear
            print curjob.name
            assert len(curjob.top) == 1
            curjob.top[0].bp()
            curjob.bp()

            #trigger the followers
            for finishedlayer in self.connections[curjob.name].bottom:
                layerstatus[finishedlayer.name] = True
            for finishedlayer in self.connections[curjob.name].bottom:
                for next_conn in finishedlayer.bp_conns:
                    if connstatus[next_conn.name] == True:
                        continue
                    connstatus[next_conn.name] = True
                    for next_conn_top in next_conn.top:
                        if layerstatus[next_conn_top.name] == False:
                            connstatus[next_conn.name] = False
                            break
                    if connstatus[next_conn.name] == True:
                        jobqueue.insert(0, next_conn)
    
    def update(self, numsamples, lr, mom, wd):
        print "======================="
        print "Update"
        for conns in self.connections:
            print conns
            if type(self.connections[conns]) == net.ConvConnection or type(self.connections[conns]) == net.FullyConnection:
                self.connections[conns].update(numsamples, lr, mom, wd)


def print_training_accuracy(o, t, minibatch_size):
    predict = o.max_index(0)
    ground_truth = t.max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)
    sys.stdout.flush()

def train_network(model, num_epochs = 100, minibatch_size=10,
        dropout_rate = 0.5, eps_w = 0.01, mom = 0.9, wd = 0.0005):
    gpu0 = owl.create_gpu_device(0)
    owl.set_device(gpu0)
    num_weights = 8
    count = 0
    last = time.time()
    cropped_size = 224

    dp = ImageNetDataProvider(mean_file='/home/minjie/data/imagenet/imagenet_mean.binaryproto',
            train_db='/home/minjie/data/imagenet/ilsvrc12_train_lmdb',
            val_db='/home/minjie/data/imagenet/ilsvrc12_val_lmdb',
            test_db='/home/minjie/data/imagenet/ilsvrc12_test_lmdb')

    #mark the output layer
    output_layer = 'prob'

    for i in xrange(num_epochs):
        print "---------------------Epoch #", i
        for (samples, labels) in dp.get_train_mb(minibatch_size, cropped_size):
            count = count + 1
            num_samples = samples.shape[0]
            data = owl.from_numpy(samples).reshape([cropped_size, cropped_size, 3, num_samples])
            target = owl.from_numpy(labels)
            model.ff(data, target)
            print_training_accuracy(model.layers[output_layer].get_act(), target, minibatch_size)
            model.bp(data, target)
            exit(0)

if __name__ == '__main__':
    owl.initialize(sys.argv)
    cpu = owl.create_cpu_device()
    owl.set_device(cpu)
    #newconfig = CaffeModelConfig(configfile = '/home/tianjun/athena/athena/owl/apps/GoogLeNet/train_val.prototxt')
    #newconfig = CaffeModelConfig(configfile = '/home/minjie/caffe/caffe/models/bvlc_reference_caffenet/train_val.prototxt')
    newconfig = CaffeModelConfig(configfile = '/home/minjie/caffe/caffe/models/VGG/VGG_train_val.prototxt')
    model = MinervaModel(newconfig.netconfig, './VGGmodel')
    train_network(model)



