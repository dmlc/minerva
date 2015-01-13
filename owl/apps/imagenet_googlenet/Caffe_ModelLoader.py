import sys,os,gc
import lmdb
import numpy as np
import numpy.random
import subprocess
from caffe_data_pb2 import Datum
from caffe_data_pb2 import BlobProto
from caffe_data_pb2 import NetParameter
from caffe_data_pb2 import LayerParameter
from PIL import Image

class CaffeModelLoader:
    def __init__(self, model_file, weightdir):
        netparam = NetParameter()
        layerparam = LayerParameter()
        with open(model_file, 'rb') as f:
            netparam.ParseFromString(f.read())
        
        cmd = 'mkdir %s' % (weightdir) 
        res = subprocess.call(cmd, shell=True)

        cmd = 'mkdir %s/epoch0' % (weightdir) 
        res = subprocess.call(cmd, shell=True)

        print len(netparam.layers)
        curweights = 0
        for i in range(len(netparam.layers)):
            #print '%d %d' % (i, curweights)
            if hasattr(netparam.layers[i], 'blobs') and len(netparam.layers[i].blobs) == 2:
                filename = '%s/epoch0/%s_weights.dat' % (weightdir, netparam.layers[i].name)
                #print filename
                if netparam.layers[i].type == layerparam.LayerType.Value('CONVOLUTION'):
                    num_output = netparam.layers[i].convolution_param.num_output
                    kernelsize = netparam.layers[i].convolution_param.kernel_size
                    orifilters = np.array(netparam.layers[i].blobs[0].data, dtype=np.float32)
                    channels = np.shape(orifilters)[0] / num_output / kernelsize / kernelsize
                    orifilters = orifilters.reshape([num_output, channels, kernelsize, kernelsize])
                    newfilters = np.zeros(np.shape(orifilters), dtype=np.float32)
                    for outidx in range(num_output):
                        for chaidx in range(channels):
                            newfilters[outidx, chaidx, :, :] = np.rot90(orifilters[outidx, chaidx, :,:],2)
                    newfilters.reshape(np.prod(np.shape(newfilters)[0:4])).tofile(filename)
                else:
                    num_output = netparam.layers[i].inner_product_param.num_output
                    input_dim = np.shape(np.array(netparam.layers[i].blobs[0].data, dtype=np.float32))[0] / num_output
                    theweight = np.transpose(np.array(netparam.layers[i].blobs[0].data, dtype=np.float32).reshape([num_output, input_dim]))
                    theweight.tofile(filename)
                #np.array(netparam.layers[i].blobs[0].data, dtype=np.float32).tofile(filename)
                
                filename = '%s/epoch0/%s_bias.dat' % (weightdir, netparam.layers[i].name)
                #print filename
                np.array(netparam.layers[i].blobs[1].data, dtype=np.float32).tofile(filename)

if __name__ == '__main__':
    dp = CaffeModelLoader(model_file='/home/minjie/caffe/caffe/models/VGG/VGG_ILSVRC_16_layers.caffemodel', weightdir = 'VGGmodel')
    count = 0
