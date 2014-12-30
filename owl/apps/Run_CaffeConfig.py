import sys,os,gc
import lmdb
import numpy as np
import numpy.random
import subprocess
import time
from caffe_pb2 import NetParameter
from caffe_pb2 import LayerParameter
from google.protobuf import text_format
import owl
from owl.conv import *
import owl.elewise as ele
from imagenet_lmdb import ImageNetDataProvider
from imagenet_lmdb_val import ImageNetDataValProvider
from PIL import Image

#read caffe config
class CaffeModelConfig:
    def __init__(self, configfile):
        file = open(configfile, "r")
        self.netconfig = NetParameter()        
        text_format.Merge(str(file.read()), self.netconfig)
        self.netconfig

#Create Minerva Model
class MinervaModel:
    def __init__(self, netconfig):
        self.num_layers = 0
        self.num_weights = 0
        self.conv2fullayer = -1
        self.weights = []
        self.weightsdelta = []
        self.bias = []
        self.biasdelta = []
        self.data = []
        layerparam = LayerParameter()
        self.ff_infos = []

        #input can be defined in layers or out of layers
        if len(netconfig.input_dim) == 4:
            self.input_size = netconfig.input_dim[3]
            self.input_channel = netconfig.input_dim[1]

        for i in range(len(netconfig.layers)):
            if netconfig.layers[i].type == layerparam.LayerType.Value('DATA'):
                self.input_size = netconfig.layers[i].transform_param.crop_size
                self.input_channel = 3
            
            elif netconfig.layers[i].type == layerparam.LayerType.Value('CONVOLUTION'):
                self.num_layers += 1
                self.ff_infos.append({'ff_type':'conv', 'neuron_type':'LINEAR', 'dropout_rate':0, 'convolution_param':netconfig.layers[i].convolution_param})
                #convinfo
                self.ff_infos[self.num_layers-1]['conv_info'] = conv_info(netconfig.layers[i].convolution_param.pad, netconfig.layers[i].convolution_param.pad, netconfig.layers[i].convolution_param.stride, netconfig.layers[i].convolution_param.stride)

                self.ff_infos[self.num_layers-1]['layer_lr'] = netconfig.layers[i].blobs_lr
                self.ff_infos[self.num_layers-1]['layer_wd'] = netconfig.layers[i].weight_decay
            
            elif netconfig.layers[i].type == layerparam.LayerType.Value('INNER_PRODUCT'):
                self.num_layers += 1
                self.ff_infos.append({'ff_type':'fully', 'neuron_type':'LINEAR', 'dropout_rate':0, 'fully_param':netconfig.layers[i].inner_product_param})
                #learning papram
                self.ff_infos[self.num_layers-1]['layer_lr'] = netconfig.layers[i].blobs_lr
                self.ff_infos[self.num_layers-1]['layer_wd'] = netconfig.layers[i].weight_decay
                
                if self.num_layers - 2 >= 0 and (self.ff_infos[self.num_layers-2]['ff_type'] == 'conv' or self.ff_infos[self.num_layers-2]['ff_type'] == 'pooling'):
                    self.conv2fullayer = self.num_layers - 1

            elif netconfig.layers[i].type == layerparam.LayerType.Value('POOLING'):
                self.num_layers += 1
                #TODO:Currently only support MAXPOOLING
                kernelsize = netconfig.layers[i].pooling_param.kernel_size
                stride = netconfig.layers[i].pooling_param.stride
                self.ff_infos.append({'ff_type':'pooling', 'pooling_info':pooling_info(kernelsize, kernelsize, stride, stride, pool_op.max), 'neuron_type':'LINEAR', 'dropout_rate':0,'pooling_param': netconfig.layers[i].pooling_param })

            elif netconfig.layers[i].type == layerparam.LayerType.Value('RELU'):
                self.ff_infos[self.num_layers-1]['neuron_type'] = 'RELU'

            elif netconfig.layers[i].type == layerparam.LayerType.Value('SOFTMAX_LOSS'):
                self.ff_infos[self.num_layers-1]['neuron_type'] = 'SOFTMAX'
            
            elif netconfig.layers[i].type == layerparam.LayerType.Value('DROPOUT'):
                self.ff_infos[self.num_layers-1]['dropout_rate'] = netconfig.layers[i].dropout_param.dropout_ratio
        
        self.num_weights = self.num_layers
        self.num_layers += 1


    def init_random(self):
        last_channel = self.input_channel
        last_scale = self.input_size
        last_dim = last_scale * last_scale * last_channel

        for i in range(self.num_weights):
            if self.ff_infos[i]['ff_type'] == 'conv':
                kernelsize = self.ff_infos[i]['convolution_param'].kernel_size
                out_channel = self.ff_infos[i]['convolution_param'].num_output
                stride = self.ff_infos[i]['convolution_param'].stride
                pad = self.ff_infos[i]['convolution_param'].pad

                print 'conv %d %d %d %d %d %d %d %d' % (i, kernelsize, out_channel, stride, pad, last_channel, last_scale, last_dim)
                owl.randn([kernelsize, kernelsize, last_channel, out_channel], 0.0, self.ff_infos[i]['convolution_param'].weight_filler.std)
                #weight
                if self.ff_infos[i]['convolution_param'].weight_filler.type == "gaussian":
                    self.weights.append(owl.randn([kernelsize, kernelsize, last_channel, out_channel], 0.0, self.ff_infos[i]['convolution_param'].weight_filler.std))
                elif self.ff_infos[i]['convolution_param'].weight_filler.type == "constant":
                    self.weights.append(owl.zeros([kernelsize, kernelsize, last_channel, out_channel]) + self.ff_infos[i]['convolution_param'].weight_filler.value)
                else:
                    assert False
                self.weightsdelta.append(owl.zeros([kernelsize, kernelsize, last_channel, out_channel]))
                
                #bias
                if self.ff_infos[i]['convolution_param'].bias_filler.type == "gaussian":
                    self.bias.append(owl.randn([out_channel], 0.0, self.ff_infos[i]['convolution_param'].bias_filler.std))
                elif self.ff_infos[i]['convolution_param'].bias_filler.type == "constant":
                    self.bias.append(owl.zeros([out_channel]) + self.ff_infos[i]['convolution_param'].bias_filler.value)
                else:
                    assert False
                self.biasdelta.append(owl.zeros([out_channel]))

                last_channel = out_channel
                last_scale = (last_scale + pad * 2 - kernelsize) / stride + 1
                last_dim = last_scale * last_scale * last_channel
            
            elif self.ff_infos[i]['ff_type'] == 'pooling':
                kernelsize = self.ff_infos[i]['pooling_param'].kernel_size
                stride = self.ff_infos[i]['pooling_param'].stride
                pad = self.ff_infos[i]['pooling_param'].pad
                print 'pool %d %d %d %d %d %d %d' % (i, kernelsize, stride, pad, last_channel, last_scale, last_dim)
                
                self.weights.append(owl.zeros([1]))
                self.weightsdelta.append(owl.zeros([1]))
                self.bias.append(owl.zeros([1]))
                self.biasdelta.append(owl.zeros([1]))
                last_channel = out_channel
                last_scale = (last_scale + pad * 2 - kernelsize) / stride + 1
                last_dim = last_scale * last_scale * last_channel
            elif self.ff_infos[i]['ff_type'] == 'fully':
                out_channel = self.ff_infos[i]['fully_param'].num_output
                
                print 'fully %d %d %d' % (i, last_dim, out_channel)
                
                #weight
                if self.ff_infos[i]['fully_param'].weight_filler.type == "gaussian":
                    self.weights.append(owl.randn([out_channel, last_dim], 0.0, self.ff_infos[i]['fully_param'].weight_filler.std))
                elif self.ff_infos[i]['fully_param'].weight_filler.type == "constant":
                    self.weights.append(owl.zeros([out_channel, last_dim]) + self.ff_infos[i]['fully_param'].weight_filler.value)
                else:
                    assert False
                self.weightsdelta.append(owl.zeros([out_channel, last_dim]))
                
                #bias
                if self.ff_infos[i]['fully_param'].bias_filler.type == "gaussian":
                    self.bias.append(owl.randn([out_channel, 1], 0.0, self.ff_infos[i]['fully_param'].weight_filler.std))
                elif self.ff_infos[i]['fully_param'].bias_filler.type == "constant":
                    self.bias.append(owl.zeros([out_channel, 1]) + self.ff_infos[i]['fully_param'].weight_filler.value)
                else:
                    assert False
                self.biasdelta.append(owl.zeros([out_channel, 1]))                 
                last_dim = out_channel
                last_channel = out_channel

#train one epoch
def train_one_mb(model, data, label, weightsgrad, biasgrad):
    #Be careful, python list is like pointer
    acts = [None] * model.num_layers
    sens = [None] * model.num_layers
    beforeacts = [None] * model.num_layers
    beforedropout = [None] * model.num_layers
    dropoutmask = [None] * model.num_layers
    before2fullyact = []
    conv2fullylayer = model.num_layers

    acts[0] = data
    num_samples = data.shape[-1]
    num_class = label.shape[0]

    #find the reshape layer

    for i in range(0, model.num_layers-1):
        if model.ff_infos[i]['ff_type'] == 'conv':
            beforeacts[i+1] = conv_forward(acts[i], model.weights[i], model.bias[i], model.ff_infos[i]['conv_info'])
        elif model.ff_infos[i]['ff_type'] == 'pooling':
            beforeacts[i+1] = pooling_forward(acts[i], model.ff_infos[i]['pooling_info'])
        else:
            beforeacts[i+1] = model.weights[i] * acts[i] + model.bias[i]

        #activation function
        if model.ff_infos[i]['neuron_type'] == 'RELU':
            acts[i+1] = ele.relu(beforeacts[i+1])
        elif model.ff_infos[i]['neuron_type'] == 'SOFTMAX':
            acts[i+1] = softmax_forward(beforeacts[i+1].reshape([num_class, 1, 1, num_samples]), soft_op.instance).reshape([num_class, num_samples]) # prob
        else:
            acts[i+1] = beforeacts[i+1]
        
        #dropout
        beforedropout[i+1] = acts[i+1]
        if model.ff_infos[i]['dropout_rate'] > 0:
            dropoutmask[i+1] = owl.randb(acts[i+1].shape, model.ff_infos[i]['dropout_rate']) 
            acts[i+1] = ele.mult(beforedropout[i+1], dropoutmask[i+1])

        if i+1 == model.conv2fullayer:
            before2fullyact = acts[i+1]
            acts[i+1] = before2fullyact.reshape([np.prod(before2fullyact.shape[0:3]), num_samples])

    # error
    sens[model.num_layers - 1] = acts[model.num_layers - 1] - label
    

    #bp
    for i in range(model.num_layers - 1, 0, -1):
        if model.ff_infos[i-1]['ff_type'] == 'conv':
            sens[i-1] = conv_backward_data(sens[i], model.weights[i-1], model.ff_infos[i-1]['conv_info'])
        elif model.ff_infos[i-1]['ff_type'] == 'pooling':
            if i == model.conv2fullayer: 
                sens[i-1] = pooling_backward(sens[i].reshape(before2fullyact.shape), before2fullyact, acts[i-1], model.ff_infos[i-1]['pooling_info'])
            else:
                sens[i-1] = pooling_backward(sens[i], acts[i], acts[i-1], model.ff_infos[i-1]['pooling_info'])
        else:
            sens[i-1] = model.weights[i-1].trans() * sens[i]

        if i - 2 >= 0:
            #dropout
            if model.ff_infos[i-2]['dropout_rate'] > 0:
                sens[i-1] = ele.mult(sens[i-1], dropoutmask[i-1])
            
            #backact
            if model.ff_infos[i-2]['neuron_type'] == 'RELU':
                sens[i-1] = ele.relu_back(sens[i-1], beforedropout[i-1], beforeacts[i-1])
            else:
                sens[i-1] = sens[i-1]

    #gradient
    for i in range(0, model.num_layers-1):
        if model.ff_infos[i]['ff_type'] == 'conv':
            weightsgrad[i] = conv_backward_filter(sens[i+1], acts[i], model.ff_infos[i]['conv_info'])
            biasgrad[i] = conv_backward_bias(sens[i+1])
        elif model.ff_infos[i]['ff_type'] == 'fully':
            weightsgrad[i] = sens[i+1] * acts[i].trans()
            biasgrad[i] = sens[i+1].sum(1)
        else:
            continue
    return acts[model.num_layers-1]

def loadmodel(i, model):
    basedir = './newinitmodel/epoch%d/' % (i)
    print 'load from %s' % (basedir)
    for k in range(model.num_layers-1):
        weightshape = model.weights[k].shape
        filename = '%sweights_%d.dat' % (basedir, k)
        weightarray = np.fromfile(filename, dtype=np.float32)
        model.weights[k] = owl.from_nparray(weightarray).reshape(weightshape)

        weightshape = model.weightsdelta[k].shape
        filename = '%sweightsdelta_%d.dat' % (basedir, k)
        weightarray = np.fromfile(filename, dtype=np.float32)
        model.weightsdelta[k] = owl.from_nparray(weightarray).reshape(weightshape)

        weightshape = model.bias[k].shape
        filename = '%sbias_%d.dat' % (basedir, k)
        weightarray = np.fromfile(filename, dtype=np.float32)
        model.bias[k] = owl.from_nparray(weightarray).reshape(weightshape)

        weightshape = model.biasdelta[k].shape
        filename = '%sbiasdelta_%d.dat' % (basedir, k)
        weightarray = np.fromfile(filename, dtype=np.float32)
        model.biasdelta[k] = owl.from_nparray(weightarray).reshape(weightshape)

def savemodel(i, model):
    basedir = './newinitmodel/epoch%d/' % (i)
    print 'save to %s' % (basedir)
    cmd = 'mkdir %s' % (basedir)
    res = subprocess.call(cmd, shell=True)
    
    for k in range(model.num_layers-1):
        weightslist = model.weights[k].tolist()
        weightarray = np.array(weightslist, dtype=np.float32)
        filename = '%sweights_%d.dat' % (basedir, k)
        weightarray.tofile(filename)
 
        weightslist = model.weightsdelta[k].tolist()
        weightarray = np.array(weightslist, dtype=np.float32)
        filename = '%sweightsdelta_%d.dat' % (basedir, k)
        weightarray.tofile(filename)       

        weightslist = model.bias[k].tolist()
        weightarray = np.array(weightslist, dtype=np.float32)
        filename = '%sbias_%d.dat' % (basedir, k)
        weightarray.tofile(filename)

        weightslist = model.biasdelta[k].tolist()
        weightarray = np.array(weightslist, dtype=np.float32)
        filename = '%sbiasdelta_%d.dat' % (basedir, k)
        weightarray.tofile(filename)

def print_training_accuracy(o, t, minibatch_size):
    predict = o.max_index(0)
    ground_truth = t.max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)
    sys.stdout.flush()
    return correct

def train_network_n(n, model, num_epochs = 100, minibatch_size=256,
        dropout_rate = 0.5, eps_w = 0.0001, eps_b = 0.0002, mom = 0.9, wd = 0.0005):
    
    gpus = []
    for i in range(0, n):
        gpus.append(owl.create_gpu_device(i))
    
    count = 0
    last = time.time()

    dp = ImageNetDataProvider(mean_file='/home/minjie/data/imagenet/imagenet_mean.binaryproto',
            train_db='/home/minjie/data/imagenet/ilsvrc12_train_lmdb',
            val_db='/home/minjie/data/imagenet/ilsvrc12_val_lmdb',
            test_db='/home/minjie/data/imagenet/ilsvrc12_test_lmdb')

    minibatch_size = minibatch_size / n
    correct = 0
    
    rerun = False
    startepoch = 0
    curepoch = startepoch

    data = [None] * n
    label = [None] * n
    out = [None] * n
    biasgrad = [None] * n
    weightsgrad = [None] * n

    #savemodel(0, model)
    #loadmodel(0, model)

    for i in range(startepoch, num_epochs):
        print "---------------------Epoch %d Index %d" % (curepoch, i)
        sys.stdout.flush()
        
        count = 0
        for (samples, labels) in dp.get_train_mb(minibatch_size):
            count = count + 1
            data[count - 1] = owl.from_nparray(samples).reshape([model.input_size, model.input_size, model.input_channel, samples.shape[0]])
            label[count - 1] = owl.from_nparray(labels)
            biasgrad[count - 1] = [None] * (model.num_layers - 1)
            weightsgrad[count - 1] = [None] * (model.num_layers - 1)
            owl.set_device(gpus[count - 1])
            out[count - 1] = train_one_mb(model, data[count - 1], label[count - 1], weightsgrad[count - 1], biasgrad[count - 1])
            out[count - 1].start_eval()
            if count % n > 0:
                continue
           
            totalweightsgrad = [None] * (model.num_layers - 1)
            totalbiasgrad = [None] * (model.num_layers - 1)
            num_samples = 0
            for gpuidx in range(0, n):
                num_samples += data[gpuidx].shape[-1]
                for k in range(model.num_layers-1):
                    if model.ff_infos[k]['ff_type'] == 'conv' or model.ff_infos[k]['ff_type'] == 'fully':
                        if gpuidx == 0:
                            totalweightsgrad[k] = weightsgrad[gpuidx][k]
                            totalbiasgrad[k] = biasgrad[gpuidx][k]
                        else:
                            totalweightsgrad[k] += weightsgrad[gpuidx][k]
                            totalbiasgrad[k] += biasgrad[gpuidx][k]

            for k in range(model.num_layers-1):
                if model.ff_infos[k]['ff_type'] == 'conv' or model.ff_infos[k]['ff_type'] == 'fully':
                    model.weightsdelta[k] = mom * model.weightsdelta[k] - model.ff_infos[k]['layer_lr'][0] * eps_w / num_samples  * (totalweightsgrad[k] + model.ff_infos[k]['layer_wd'][0] * wd * num_samples * model.weights[k])
                    model.biasdelta[k] = mom * model.biasdelta[k] -  model.ff_infos[k]['layer_lr'][1] * eps_b / num_samples  * (totalbiasgrad[k] + model.ff_infos[k]['layer_wd'][1] * wd * num_samples * model.bias[k])
                    model.weights[k] += model.weightsdelta[k]
                    model.bias[k] += model.biasdelta[k]
        
            #print num_samples
            if count % n == 0:
                print 'count %d' % (count)
                thiscorrect = print_training_accuracy(out[0], label[0], data[0].shape[-1])
                print "time: %s" % (time.time() - last)
                last = time.time()
                count = 0



if __name__ == '__main__':
    owl.initialize(sys.argv)
    cpu = owl.create_cpu_device()
    owl.set_device(cpu)
    newconfig = CaffeModelConfig(configfile = '/home/minjie/caffe/caffe/models/bvlc_reference_caffenet/train_val.prototxt')
    #newconfig = CaffeModelConfig(configfile = '/home/minjie/caffe/caffe/models/VGG/VGG_ILSVRC_16_layers_deploy.prototxt')
    
    model = MinervaModel(newconfig.netconfig)
    model.init_random()
    train_network_n(2, model)

