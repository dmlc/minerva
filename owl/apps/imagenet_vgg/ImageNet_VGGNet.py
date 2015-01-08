import math
import sys
import time
import numpy as np
import subprocess

import owl
from owl.conv import *
import owl.elewise as ele
from imagenet_lmdb import ImageNetDataProvider
from imagenet_lmdb_val import ImageNetDataValProvider
from PIL import Image

class VGGModel:
    def __init__(self):
        self.num_layers = 22 
        self.weights = []
        self.weightsdelta = []
        self.bias = []
        self.data = []
        self.ff_infos = [
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'pooling', 'pooling_info': pooling_info(2,2,2,2, pool_op.max), 'neuron_type':'LINEAR', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'pooling', 'pooling_info': pooling_info(2,2,2,2, pool_op.max), 'neuron_type':'LINEAR', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'pooling', 'pooling_info': pooling_info(2,2,2,2, pool_op.max), 'neuron_type':'LINEAR', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'pooling', 'pooling_info': pooling_info(2,2,2,2, pool_op.max), 'neuron_type':'LINEAR', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'conv', 'conv_info': conv_info(1,1,1,1), 'neuron_type':'RELU', 'dropout_rate': 0},
                {'ff_type': 'pooling', 'pooling_info': pooling_info(2,2,2,2, pool_op.max), 'neuron_type':'LINEAR', 'dropout_rate': 0},
                {'ff_type': 'fully', 'neuron_type':'RELU', 'dropout_rate':0.5}, 
                {'ff_type': 'fully', 'neuron_type':'RELU', 'dropout_rate':0.5},
                {'ff_type': 'fully', 'neuron_type':'SOFTMAX', 'dropout_rate':0} 
        ];
    def init_random(self):
        self.weights = [
            owl.randn([3, 3, 3, 64], 0.0, 0.01),
            owl.randn([3, 3, 64, 64], 0.0, 0.01),
            owl.randn([1, 1, 1, 1], 0.0, 0.01),
            owl.randn([3, 3, 64, 128], 0.0, 0.01),
            owl.randn([3, 3, 128, 128], 0.0, 0.01),
            owl.randn([1, 1, 1, 1], 0.0, 0.01),
            owl.randn([3, 3, 128, 256], 0.0, 0.01),
            owl.randn([3, 3, 256, 256], 0.0, 0.01),
            owl.randn([3, 3, 256, 256], 0.0, 0.01),
            owl.randn([1, 1, 1, 1], 0.0, 0.01),
            owl.randn([3, 3, 256, 512], 0.0, 0.01),
            owl.randn([3, 3, 512, 512], 0.0, 0.01),
            owl.randn([3, 3, 512, 512], 0.0, 0.01),
            owl.randn([1, 1, 1, 1], 0.0, 0.01),
            owl.randn([3, 3, 512, 512], 0.0, 0.01),
            owl.randn([3, 3, 512, 512], 0.0, 0.01),
            owl.randn([3, 3, 512, 512], 0.0, 0.01),
            owl.randn([1, 1, 1, 1], 0.0, 0.01),
            owl.randn([4096, 25088], 0.0, 0.005),
            owl.randn([4096, 4096], 0.0, 0.005),
            owl.randn([1000, 4096], 0.0, 0.01)
        ];
        self.weightsdelta = [
            owl.zeros([3, 3, 3, 64]),
            owl.zeros([3, 3, 64, 64]),
            owl.zeros([1, 1, 1, 1]),
            owl.zeros([3, 3, 64, 128]),
            owl.zeros([3, 3, 128, 128]),
            owl.zeros([1, 1, 1, 1]),
            owl.zeros([3, 3, 128, 256]),
            owl.zeros([3, 3, 256, 256]),
            owl.zeros([3, 3, 256, 256]),
            owl.zeros([1, 1, 1, 1]),
            owl.zeros([3, 3, 256, 512]),
            owl.zeros([3, 3, 512, 512]),
            owl.zeros([3, 3, 512, 512]),
            owl.zeros([1, 1, 1, 1]),
            owl.zeros([3, 3, 512, 512]),
            owl.zeros([3, 3, 512, 512]),
            owl.zeros([3, 3, 512, 512]),
            owl.zeros([1, 1, 1, 1]),
            owl.zeros([4096, 25088]),
            owl.zeros([4096, 4096]),
            owl.zeros([1000, 4096])
        ];

        self.bias = [
            owl.zeros([64]),
            owl.zeros([64]),
            owl.zeros([64]),
            owl.zeros([128]),
            owl.zeros([128]),
            owl.zeros([128]),
            owl.zeros([256]),
            owl.zeros([256]),
            owl.zeros([256]),
            owl.zeros([256]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([4096, 1]),
            owl.zeros([4096, 1]),
            owl.zeros([1000, 1])
        ];
        
        self.biasdelta = [
            owl.zeros([64]),
            owl.zeros([64]),
            owl.zeros([64]),
            owl.zeros([128]),
            owl.zeros([128]),
            owl.zeros([128]),
            owl.zeros([256]),
            owl.zeros([256]),
            owl.zeros([256]),
            owl.zeros([256]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([512]),
            owl.zeros([4096, 1]),
            owl.zeros([4096, 1]),
            owl.zeros([1000, 1])
        ];


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
        #if from conv 2 fully
        if (i < model.num_layers -2) and (model.ff_infos[i]['ff_type'] == 'conv' or model.ff_infos[i]['ff_type'] == 'pooling') and (model.ff_infos[i+1]['ff_type'] == 'fully'):
            conv2fullylayer = i + 1
            break

    for i in range(0, model.num_layers-1):
        if model.ff_infos[i]['ff_type'] == 'conv':
            #print '%d conv ff' % (i)
            beforeacts[i+1] = conv_forward(acts[i], model.weights[i], model.bias[i], model.ff_infos[i]['conv_info'])
        elif model.ff_infos[i]['ff_type'] == 'pooling':
            #print '%d pooling ff' % (i)
            beforeacts[i+1] = pooling_forward(acts[i], model.ff_infos[i]['pooling_info'])
        else:
            #print '%d fully ff' % (i)
            beforeacts[i+1] = model.weights[i] * acts[i] + model.bias[i]

        #activation function
        if model.ff_infos[i]['neuron_type'] == 'RELU':
            #print '%d relu ff' % (i)
            acts[i+1] = ele.relu(beforeacts[i+1])
        elif model.ff_infos[i]['neuron_type'] == 'SOFTMAX':
            #print '%d softmax ff' % (i)
            acts[i+1] = softmax_forward(beforeacts[i+1].reshape([num_class, 1, 1, num_samples]), soft_op.instance).reshape([num_class, num_samples]) # prob
        else:
            #print '%d linear ff' % (i)
            acts[i+1] = beforeacts[i+1]
        
        #dropout
        beforedropout[i+1] = acts[i+1]
        if model.ff_infos[i]['dropout_rate'] > 0:
            #print '%d dropout ff' % (i)
            dropoutmask[i+1] = owl.randb(acts[i+1].shape, model.ff_infos[i]['dropout_rate']) 
            acts[i+1] = ele.mult(beforedropout[i+1], dropoutmask[i+1])

        if i+1 == conv2fullylayer:
            before2fullyact = acts[i+1]
            acts[i+1] = before2fullyact.reshape([np.prod(before2fullyact.shape[0:3]), num_samples])

    # error
    sens[model.num_layers - 1] = acts[model.num_layers - 1] - label

    #bp
    for i in range(model.num_layers - 1, 0, -1):
        if model.ff_infos[i-1]['ff_type'] == 'conv':
            sens[i-1] = conv_backward_data(sens[i], model.weights[i-1], model.ff_infos[i-1]['conv_info'])
        elif model.ff_infos[i-1]['ff_type'] == 'pooling':
            if i == conv2fullylayer: 
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
    basedir = './VGGmodel/epoch%d/' % (i)
    print 'load from %s' % (basedir)
    for k in range(model.num_layers-1):
        weightshape = model.weights[k].shape
        filename = '%sweights_%d.dat' % (basedir, k)
        weightarray = np.fromfile(filename, dtype=np.float32)
        model.weights[k] = owl.from_numpy(weightarray).reshape(weightshape)

        weightshape = model.weightsdelta[k].shape
        filename = '%sweightsdelta_%d.dat' % (basedir, k)
        weightarray = np.fromfile(filename, dtype=np.float32)
        model.weightsdelta[k] = owl.from_numpy(weightarray).reshape(weightshape)

        weightshape = model.bias[k].shape
        filename = '%sbias_%d.dat' % (basedir, k)
        weightarray = np.fromfile(filename, dtype=np.float32)
        model.bias[k] = owl.from_numpy(weightarray).reshape(weightshape)

        weightshape = model.biasdelta[k].shape
        filename = '%sbiasdelta_%d.dat' % (basedir, k)
        weightarray = np.fromfile(filename, dtype=np.float32)
        model.biasdelta[k] = owl.from_numpy(weightarray).reshape(weightshape)

def savemodel(i, model):
    basedir = './VGGmodel/epoch%d/' % (i)
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
    
    '''
    print predict.tolist()
    print ground_truth.tolist()
    print 'hah'
    '''

    correct = (predict - ground_truth).count_zero()
    print "%d %d" % (minibatch_size, correct)
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)
    sys.stdout.flush()
    return correct

def train_network_n(n, model, num_epochs = 100, minibatch_size=40,
        dropout_rate = 0.5, eps_w = 0.0001, eps_b = 0.0002, mom = 0.9, wd = 0.0005):
    
    gpus = []
    for i in range(0, n):
        gpus.append(owl.create_gpu_device(i))
    
    count = 0
    last = time.time()

    dp = ImageNetDataProvider(mean_file='./VGGmodel/vgg_mean.binaryproto',
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


    for i in range(startepoch, num_epochs):
        print "---------------------Epoch %d Index %d" % (curepoch, i)
        sys.stdout.flush()
        batchidx = 0
        count = 0
        loadmodel(i, model)
        for (samples, labels) in dp.get_train_mb(minibatch_size, 224):
            count = count + 1
            data[count - 1] = owl.from_numpy(samples).reshape([224, 224, 3, samples.shape[0]])
            label[count - 1] = owl.from_numpy(labels)
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
                    model.weightsdelta[k] = mom * model.weightsdelta[k] - eps_w / num_samples  * (totalweightsgrad[k] + wd * num_samples * model.weights[k])
                    model.biasdelta[k] = mom * model.biasdelta[k] - eps_b / num_samples  * totalbiasgrad[k]
                    model.weights[k] += model.weightsdelta[k]
                    model.bias[k] += model.biasdelta[k]
        
            #print num_samples
            if count % n == 0:
                print 'batch %d' % (batchidx)
                batchidx = batchidx + 1
                
                '''
                #TODO hack
                if batchidx == 2000:
                    savemodel(i+1, model)
                    exit(0)
                '''
                thiscorrect = print_training_accuracy(out[0], label[0], data[0].shape[-1])
                print "time: %s" % (time.time() - last)
                last = time.time()
                count = 0
        savemodel(i+1, model)

if __name__ == '__main__':
    owl.initialize(sys.argv)
    cpu = owl.create_cpu_device()
    owl.set_device(cpu)
    model = VGGModel()
    model.init_random()
    train_network_n(2, model)





