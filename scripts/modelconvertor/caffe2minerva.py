import os
import sys
import owl
from owl.net.caffe import *
from google.protobuf import text_format
import numpy as np
import owl
import subprocess

class Caffe2MinervaConvertor:
    ''' Class to convert Caffe's caffemodel into numpy array files. Minerva use numpy array files to store and save model snapshots.
    :ivar str model_file: Caffe's caffemodel
    :ivar str weightdir: directory to save numpy-array models
    :ivar int snapshot: snapshot index
    '''
    def __init__(self, model_file, weightdir, snapshot):
        netparam = NetParameter()
        layerparam = V1LayerParameter()
        with open(model_file, 'rb') as f:
            netparam.ParseFromString(f.read())

        cmd = 'mkdir %s' % (weightdir) 
        res = subprocess.call(cmd, shell=True)

        cmd = 'mkdir %s/snapshot%d' % (weightdir, snapshot) 
        res = subprocess.call(cmd, shell=True)

        curweights = 0
        for i in range(len(netparam.layers)):
            #print '%d %d' % (i, curweights)
            if hasattr(netparam.layers[i], 'blobs') and len(netparam.layers[i].blobs) == 2:
                layername = netparam.layers[i].name
                layername = layername.replace("/","_")
                filename = '%s/snapshot%d/%s_weights.dat' % (weightdir, snapshot, layername)
                print filename
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
                    print netparam.layers[i].blobs[0]
                    exit(0)

                filename = '%s/snapshot%d/%s_bias.dat' % (weightdir, snapshot, layername)
                print filename
                np.array(netparam.layers[i].blobs[1].data, dtype=np.float32).tofile(filename)
                

if __name__ == "__main__":
    Caffe2MinervaConvertor('/home/tianjun/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel', '/home/tianjun/models/newGoogmodel_script/', 0)
 






