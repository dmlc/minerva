import math
import sys
import time
import numpy as np

import owl
from owl.conv import *
import owl.elewise as ele
from imagenet_lmdb import ImageNetDataProvider

class AlexModel:
    def __init__(self):
        self.weights = []
	self.weightsdelta = []
        self.bias = []
        self.biasdelta = []
	self.conv_infos = [
            conv_info(0, 0, 4, 4), # conv1
            conv_info(2, 2, 1, 1), # conv2
            conv_info(1, 1, 1, 1), # conv3
            conv_info(1, 1, 1, 1), # conv4
            conv_info(1, 1, 1, 1)  # conv5
        ];
        self.pooling_infos = [
            pooling_info(3, 3, 2, 2, pool_op.max), # pool1
            pooling_info(3, 3, 2, 2, pool_op.max), # pool2
            pooling_info(3, 3, 2, 2, pool_op.max)  # pool5
        ];

    def init_random(self):
        self.weights = [
            owl.randn([11, 11, 3, 96], 0.0, 0.01),
            owl.randn([5, 5, 96, 256], 0.0, 0.01),
            owl.randn([3, 3, 256, 384], 0.0, 0.01),
            owl.randn([3, 3, 384, 384], 0.0, 0.01),
            owl.randn([3, 3, 384, 256], 0.0, 0.01),
            owl.randn([4096, 9216], 0.0, 0.01),
            owl.randn([4096, 4096], 0.0, 0.01),
            owl.randn([1000, 4096], 0.0, 0.01)
        ];

	self.weightsdelta = [
            owl.zeros([11, 11, 3, 96]),
            owl.zeros([5, 5, 96, 256]),
            owl.zeros([3, 3, 256, 384]),
            owl.zeros([3, 3, 384, 384]),
            owl.zeros([3, 3, 384, 256]),
            owl.zeros([4096, 9216]),
            owl.zeros([4096, 4096]),
            owl.zeros([1000, 4096])
        ];


        self.bias = [
            owl.zeros([96]),
            owl.zeros([256]),
            owl.zeros([384]),
            owl.zeros([384]),
            owl.zeros([256]),
            owl.zeros([4096, 1]),
            owl.zeros([4096, 1]),
            owl.zeros([1000, 1])
        ];

        self.biasdelta = [
            owl.zeros([96]),
            owl.zeros([256]),
            owl.zeros([384]),
            owl.zeros([384]),
            owl.zeros([256]),
            owl.zeros([4096, 1]),
            owl.zeros([4096, 1]),
            owl.zeros([1000, 1])
        ];

def print_training_accuracy(o, t, minibatch_size):
    #print np.array(o.tolist()).reshape([minibatch_size, 1000])
    predict = o.max_index(0)
    #print np.array(predict.tolist())
    ground_truth = t.max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)

def train_network(model, num_epochs = 100, minibatch_size=256,
        dropout_rate = 0.5, eps_w = 0.01, eps_b = 0.01, mom = 0.75, wd = 0.005):
    gpu = owl.create_gpu_device(0)
    owl.set_device(gpu)
    num_layers = 20
    count = 0
    last = time.time()

    dp = ImageNetDataProvider(mean_file='/home/minjie/data/imagenet/imagenet_mean.binaryproto',
            train_db='/home/minjie/data/imagenet/ilsvrc12_train_lmdb',
            val_db='/home/minjie/data/imagenet/ilsvrc12_val_lmdb',
            test_db='/home/minjie/data/imagenet/ilsvrc12_test_lmdb')

    acts = [None] * num_layers
    sens = [None] * num_layers

    for i in xrange(num_epochs):
        print '--------------Epoch', i
        #num_samples = 256
        #acts[0] = owl.randn([227,227,3,num_samples], 0.0, 1) * 128
        #target = owl.randn([1000,num_samples], 0.0, 1)
        #for (s, l) in dp.get_train_mb(minibatch_size):
            #samples = s
            #labels = l
            #break
        #for j in xrange(100):
        for (samples, labels) in dp.get_train_mb(minibatch_size):
            num_samples = samples.shape[0]

            # FF
            acts[0] = owl.from_nparray(samples).reshape([227, 227, 3, num_samples])
            target = owl.from_nparray(labels)

            #np.set_printoptions(linewidth=200)
            #print acts[0].shape, model.weights[0].shape, model.bias[0].shape
            #im = np.array(acts[0].tolist()).reshape([num_samples, 227, 227, 3])
            #print im[0,:,:,0]
            #print im[0,:,:,1]
            #print im[0,:,:,2]
            #print target.max_index(0).tolist()[0:20]
            #sys.exit()

            acts1 = conv_forward(acts[0], model.weights[0], model.bias[0], model.conv_infos[0])
            acts[1] = ele.relu(acts1)#(conv_forward(acts[0], model.weights[0], model.bias[0], model.conv_infos[0])) # conv1
            acts[2] = pooling_forward(acts[1], model.pooling_infos[0]) # pool1
            acts3 = conv_forward(acts[2], model.weights[1], model.bias[1], model.conv_infos[1]) # conv2
            acts[3] = ele.relu(acts3)#(conv_forward(acts[2], model.weights[1], model.bias[1], model.conv_infos[1])) # conv2
            acts[4] = pooling_forward(acts[3], model.pooling_infos[1]) # pool2
            acts5 = conv_forward(acts[4], model.weights[2], model.bias[2], model.conv_infos[2]) # conv3
            acts[5] = ele.relu(acts5)#(conv_forward(acts[4], model.weights[2], model.bias[2], model.conv_infos[2])) # conv3
            acts6 = conv_forward(acts[5], model.weights[3], model.bias[3], model.conv_infos[3]) # conv4
            acts[6] = ele.relu(acts6)#(conv_forward(acts[5], model.weights[3], model.bias[3], model.conv_infos[3])) # conv4
            acts7 = conv_forward(acts[6], model.weights[4], model.bias[4], model.conv_infos[4]) # conv5
            acts[7] = ele.relu(acts7)#(conv_forward(acts[6], model.weights[4], model.bias[4], model.conv_infos[4])) # conv5
            acts[8] = pooling_forward(acts[7], model.pooling_infos[2]) # pool5
            re_acts8 = acts[8].reshape([np.prod(acts[8].shape[0:3]), num_samples])
            acts9 = model.weights[5] * re_acts8 + model.bias[5] # fc6
            acts[9] = ele.relu(acts9)#(model.weights[5] * re_acts8 + model.bias[5]) # fc6
            mask6 = owl.randb(acts[9].shape, dropout_rate)
            acts[9] = ele.mult(acts[9], mask6) # drop6
            acts10 = model.weights[6] * acts[9] + model.bias[6] # fc7
            acts[10] = ele.relu(acts10)#(model.weights[6] * acts[9] + model.bias[6]) # fc7
            mask7 = owl.randb(acts[10].shape, dropout_rate)
            acts[10] = ele.mult(acts[10], mask7) # drop7
            acts[11] = model.weights[7] * acts[10] + model.bias[7] # fc8
            acts[12] = softmax_forward(acts[11].reshape([1000, 1, 1, num_samples]), soft_op.instance).reshape([1000, num_samples]) # prob

            # error
            sens[11] = acts[12] - target

            # BP
            sens[10] = model.weights[7].trans() * sens[11] # fc8
            sens[10] = ele.mult(sens[10], mask7) # drop7
            sens[10] = ele.relu_back(sens[10], acts[10], acts10) # relu7
            sens[9] = model.weights[6].trans() * sens[10]
            sens[9] = ele.mult(sens[9], mask6) # drop6
            sens[9] = ele.relu_back(sens[9], acts[9], acts9) # relu6
            sens[8] = (model.weights[5].trans() * sens[9]).reshape(acts[8].shape) # fc6
            sens[7] = pooling_backward(sens[8], acts[8], acts[7], model.pooling_infos[2]) # pool5
            sens[7] = ele.relu_back(sens[7], acts[7], acts7) # relu5
            sens[6] = conv_backward_data(sens[7], model.weights[4], model.conv_infos[4]) # conv5
            sens[6] = ele.relu_back(sens[6], acts[6], acts6) # relu4
            sens[5] = conv_backward_data(sens[6], model.weights[3], model.conv_infos[3]) # conv4
            sens[5] = ele.relu_back(sens[5], acts[5], acts5) # relu3
            sens[4] = conv_backward_data(sens[5], model.weights[2], model.conv_infos[2]) # conv3
            sens[3] = pooling_backward(sens[4], acts[4], acts[3], model.pooling_infos[1]) # pool2
            sens[3] = ele.relu_back(sens[3], acts[3], acts3) # relu2
            sens[2] = conv_backward_data(sens[3], model.weights[1], model.conv_infos[1]) # conv2
            sens[1] = pooling_backward(sens[2], acts[2], acts[1], model.pooling_infos[0]) # pool1
            sens[1] = ele.relu_back(sens[1], acts[1], acts1) # relu1

	    model.weightsdelta[7] = mom * model.weightsdelta[7] - eps_w / num_samples  * (sens[11] * acts[10].trans() + wd * model.weights[7])
            model.biasdelta[7] = mom * model.biasdelta[7] - eps_b / num_samples  * (sens[11].sum(1) + wd * model.bias[7])
            
	    model.weightsdelta[6] = mom * model.weightsdelta[6] - eps_w / num_samples  * (sens[10] * acts[9].trans() + wd * model.weights[6])
            model.biasdelta[6] = mom * model.biasdelta[6] - eps_b / num_samples  * (sens[10].sum(1) + wd * model.bias[6])
    	    
	    model.weightsdelta[5] = mom * model.weightsdelta[5] - eps_w / num_samples  * (sens[9] * re_acts8.trans() + wd * model.weights[5])
            model.biasdelta[5] = mom * model.biasdelta[5] - eps_b / num_samples  * (sens[9].sum(1) + wd * model.bias[5])
            	
            model.weightsdelta[4] = mom * model.weightsdelta[4] - eps_w / num_samples  * (conv_backward_filter(sens[7], acts[6], model.conv_infos[4]) + wd * model.weights[4])
	    model.biasdelta[4] = mom * model.biasdelta[4] - eps_b / num_samples  * (conv_backward_bias(sens[7]) + wd * model.bias[4])

	    model.weightsdelta[3] = mom * model.weightsdelta[3] - eps_w / num_samples  * (conv_backward_filter(sens[6], acts[5], model.conv_infos[3]) + wd * model.weights[3])
	    model.biasdelta[3] = mom * model.biasdelta[3] - eps_b / num_samples  * (conv_backward_bias(sens[6]) + wd * model.bias[3])

 	    model.weightsdelta[2] = mom * model.weightsdelta[2] - eps_w / num_samples  * (conv_backward_filter(sens[5], acts[4], model.conv_infos[2]) + wd * model.weights[2])
	    model.biasdelta[2] = mom * model.biasdelta[2] - eps_b / num_samples  * (conv_backward_bias(sens[5]) + wd * model.bias[2])

  	    model.weightsdelta[1] = mom * model.weightsdelta[1] - eps_w / num_samples  * (conv_backward_filter(sens[3], acts[2], model.conv_infos[1]) + wd * model.weights[1])
	    model.biasdelta[1] = mom * model.biasdelta[1] - eps_b / num_samples  * (conv_backward_bias(sens[3]) + wd * model.bias[1])

            model.weightsdelta[0] = mom * model.weightsdelta[0] - eps_w / num_samples  * (conv_backward_filter(sens[1], acts[0], model.conv_infos[0]) + wd * model.weights[0])
	    model.biasdelta[0] = mom * model.biasdelta[0] - eps_b / num_samples  * (conv_backward_bias(sens[1]) + wd * model.bias[0])

            for k in range(8):
                model.weights[k] += model.weightsdelta[k]
                model.bias[k] += model.biasdelta[k]

            count = count + 1
            #if count % 2 == 0:
                #acts[18].start_eval()
            if count % 10 == 0:
                print_training_accuracy(acts[12], target, num_samples)
                print "time: %s" % (time.time() - last)
                last = time.time()

if __name__ == '__main__':
    owl.initialize(sys.argv)
    cpu = owl.create_cpu_device()
    owl.set_device(cpu)
    model = AlexModel()
    model.init_random()
    train_network(model)

