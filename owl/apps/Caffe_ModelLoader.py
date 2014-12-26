import sys,os,gc
import lmdb
import numpy as np
import numpy.random
import subprocess
from caffe_data_pb2 import Datum
from caffe_data_pb2 import BlobProto
from caffe_data_pb2 import NetParameter
from PIL import Image

class CaffeModelLoader:
    def __init__(self, model_file, weightdir):
        netparam = NetParameter()
        with open(model_file, 'rb') as f:
            netparam.ParseFromString(f.read())

        cmd = 'mkdir %s/epoch0' % (weightdir) 
        res = subprocess.call(cmd, shell=True)

        print len(netparam.layers)
        curweights = 0
        for i in range(len(netparam.layers)):
            print '%d %d' % (i, curweights)
            if hasattr(netparam.layers[i], 'blobs') and len(netparam.layers[i].blobs) == 2:
                filename = '%s/epoch0/weights_%d.dat' % (weightdir, curweights)
                print filename
                np.array(netparam.layers[i].blobs[0].data, dtype=np.float32).tofile(filename)
                filename = '%s/epoch0/bias_%d.dat' % (weightdir, curweights)
                print filename
                np.array(netparam.layers[i].blobs[1].data, dtype=np.float32).tofile(filename)
                curweights += 1
            elif netparam.layers[i].type == layerparam.LayerType.Value('POOLING'):
                curweights += 1

if __name__ == '__main__':
    dp = CaffeModelLoader(model_file='/home/minjie/caffe/caffe/models/VGG/VGG_ILSVRC_16_layers.caffemodel', weightdir = 'VGGmodel')
    count = 0
