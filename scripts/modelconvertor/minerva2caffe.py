#!/usr/bin/env python

import os
import sys, argparse
import owl
from owl.net.caffe import *
from google.protobuf import text_format
import numpy as np
import owl
import subprocess

class Minerva2CaffeConvertor:
    def __init__(self, config_file, weightdir, snapshot, caffemodelpath):
        layerparam = V1LayerParameter()
        netparam = NetParameter()
        with open(config_file, 'r') as f:
            text_format.Merge(str(f.read()), netparam)
        
        for i in range(len(netparam.layers)):
            layername = netparam.layers[i].name
            layername = layername.replace("/","_")
            if netparam.layers[i].type == layerparam.LayerType.Value('CONVOLUTION'):
                filename = '%s/snapshot%d/%s_weights.dat' % (weightdir, snapshot, layername)
                print filename

                num_output = netparam.layers[i].convolution_param.num_output
                kernelsize = netparam.layers[i].convolution_param.kernel_size
                orifilters =  np.fromfile(filename, dtype=np.float32) 
                channels = np.shape(orifilters)[0] / num_output / kernelsize / kernelsize
                orifilters = orifilters.reshape([num_output, channels, kernelsize, kernelsize])
                newfilters = np.zeros(np.shape(orifilters), dtype=np.float32)
                for outidx in range(num_output):
                    for chaidx in range(channels):
                        newfilters[outidx, chaidx, :, :] = np.rot90(orifilters[outidx, chaidx, :,:],2)
                newfilters = newfilters.reshape(np.prod(np.shape(newfilters)[0:4]))

                thisblob = netparam.layers[i].blobs.add()
                thisblob.data.extend(newfilters.tolist())
                thisblob.num = num_output
                thisblob.channels = channels
                thisblob.height = kernelsize
                thisblob.width = kernelsize

                filename = '%s/snapshot%d/%s_bias.dat' % (weightdir, snapshot, layername)                
                theweight = np.fromfile(filename, dtype=np.float32)
                thisblob = netparam.layers[i].blobs.add()
                thisblob.data.extend(theweight.tolist())
                thisblob.num = 1
                thisblob.channels = 1
                thisblob.height = 1
                thisblob.width = num_output             
                
            elif netparam.layers[i].type == layerparam.LayerType.Value('INNER_PRODUCT'):
                filename = '%s/snapshot%d/%s_weights.dat' % (weightdir, snapshot, layername)
                print filename
                
                num_output = netparam.layers[i].inner_product_param.num_output
                orifilters = np.fromfile(filename,dtype=np.float32)
                input_dim = np.shape(orifilters)[0] / num_output
                theweight = np.transpose(orifilters.reshape([input_dim, num_output])).reshape([num_output * input_dim])
               
                thisblob = netparam.layers[i].blobs.add()
                thisblob.data.extend(theweight.tolist())
                thisblob.num = 1
                thisblob.channels = 1
                thisblob.height = num_output
                thisblob.width = input_dim
               
                filename = '%s/snapshot%d/%s_bias.dat' % (weightdir, snapshot, layername)                
                print filename
                theweight = np.fromfile(filename, dtype=np.float32)
                thisblob = netparam.layers[i].blobs.add()
                thisblob.data.extend(theweight.tolist())
                thisblob.num = 1
                thisblob.channels = 1
                thisblob.height = 1
                thisblob.width = num_output
        resultfile = open(caffemodelpath, 'w')
        resultfile.write(netparam.SerializeToString())
        resultfile.close()
    

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='caffe configure file')
    parser.add_argument('minerva_model_dir', help='minerva model dir')
    parser.add_argument('snapshot', help='the model snapshot to be converted', type=int, default=0)
    parser.add_argument('caffe_model', help='path of the converted caffe model')
    
    (args, remain) = parser.parse_known_args()
    Minerva2CaffeConvertor(args.config_file, args.minerva_model_dir, args.snapshot, args.caffe_model)
 



