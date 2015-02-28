import sys,os,gc
import lmdb
import numpy as np
import numpy.random
import subprocess
import time
from caffe_data_pb2 import NetParameter
from caffe_data_pb2 import LayerParameter
from google.protobuf import text_format
import owl
from owl.conv import *
import owl.elewise as ele
from imagenet_lmdb import ImageNetDataProvider
from imagenet_lmdb_val import ImageNetDataValProvider
from PIL import Image

#ITER ib connections, finish one connection, put the ones can be executed into the queue

class CaffeModelConfig:
    def __init__(self, configfile):
        file = open(configfile, "r")
        self.netconfig = NetParameter()        
        text_format.Merge(str(file.read()), self.netconfig)
        self.netconfig

class Component:
    def __init__(self, name):
        self.name = name
        self.ff_connect = []
        self.bp_connect = []
        self.type = 'LINEAR'
        self.droprate = 0

class Connection:
    def __init__(self, layerinfo):
        #info
        self.bottom = []
        self.top = []
        self.name = layerinfo.name
        #get the bottom and top
        for botidx in range(len(layerinfo.bottom)):
            self.bottom.append(layerinfo.bottom[botidx])
        for topidx in range(len(layerinfo.top)):
            self.top.append(layerinfo.top[topidx])
    def ff(self):
        print "Excecuting Connection: %s" % self.name 

class MinervaModel:
    def __init__(self, netconfig):
        self.components = dict()
        self.connections = dict()
        
        self.inputcomp = []
        self.outputcomp = []

        self.conv2fullayer = -1
        layerparam = LayerParameter()
  
        #input can be defined in layers or out of layers
        if len(netconfig.input_dim) == 4:
            self.input_size = netconfig.input_dim[3]
            self.input_channel = netconfig.input_dim[1]
        
        for i in range(len(netconfig.layers)):
            #data is a input layer
            if netconfig.layers[i].type == layerparam.LayerType.Value('DATA'):
                self.input_size = netconfig.layers[i].transform_param.crop_size
                self.input_channel = 3
                for topidx in range(len(netconfig.layers[i].top)):
                    if netconfig.layers[i].top[topidx] not in self.inputcomp: 
                        name = netconfig.layers[i].top[topidx]
                        self.components[name] = Component(name)
                        self.inputcomp.append(name)

            #if top[0] != bottom[0], it should be a connection, otherwise, we just consider it as a non-linear operations
            if len(netconfig.layers[i].top) >= 1 and  len(netconfig.layers[i].bottom) >= 1:
                #create connection
                if netconfig.layers[i].top[0] != netconfig.layers[i].bottom[0]:
                    print "Connection %d" % (i)
                    print netconfig.layers[i].name
                    
                    #decide connection type
                    if netconfig.layers[i].type == layerparam.LayerType.Value('CONVOLUTION'):
                        #TODO: may turn to heir class
                        self.connections[netconfig.layers[i].name] = Connection(netconfig.layers[i])
                        self.connections[netconfig.layers[i].name].type = 'convolution'
                    elif  netconfig.layers[i].type == layerparam.LayerType.Value('POOLING'):
                        self.connections[netconfig.layers[i].name] = Connection(netconfig.layers[i])
                        self.connections[netconfig.layers[i].name].type = 'pooling'
                    elif  netconfig.layers[i].type == layerparam.LayerType.Value('INNER_PRODUCT'):
                        self.connections[netconfig.layers[i].name] = Connection(netconfig.layers[i])
                        self.connections[netconfig.layers[i].name].type = 'fully'
                    elif  netconfig.layers[i].type == layerparam.LayerType.Value('LRN'):
                        self.connections[netconfig.layers[i].name] = Connection(netconfig.layers[i])
                        self.connections[netconfig.layers[i].name].type = 'lrn'
                    elif netconfig.layers[i].type == layerparam.LayerType.Value('SOFTMAX_LOSS'):
                        self.connections[netconfig.layers[i].name] = Connection(netconfig.layers[i])
                        self.components[netconfig.layers[i].bottom[0]].type = 'softmax'
                    elif netconfig.layers[i].type == layerparam.LayerType.Value('CONCAT'):
                        self.connections[netconfig.layers[i].name] = Connection(netconfig.layers[i])
                        self.components[netconfig.layers[i].bottom[0]].type = 'concat'
                    else:
                        print 'Not Implemented Connection'
                        print netconfig.layers[i].name
                        continue
 
                    #create components and it's name
                    for topidx in range(len(netconfig.layers[i].top)):
                        name = netconfig.layers[i].top[topidx]
                        self.components[name] = Component(name)                   
                    #output connection
                    for botidx in range(len(netconfig.layers[i].bottom)):
                        assert netconfig.layers[i].bottom[botidx] in self.components
                        self.components[netconfig.layers[i].bottom[botidx]].ff_connect.append(netconfig.layers[i].name)
                    assert len(netconfig.layers[i].top) == 1
                    self.components[netconfig.layers[i].top[0]].bp_connect.append(self.connections[netconfig.layers[i].name])

                #other operations just seen as the non-linear operation
                elif len(netconfig.layers[i].top) == 1 and len(netconfig.layers[i].bottom) == 1 and netconfig.layers[i].top[0] == netconfig.layers[i].bottom[0]:
                    print "Neuron %d" % (i)
                    print netconfig.layers[i].name
                    
                    assert netconfig.layers[i].bottom[0] in self.components
                    if netconfig.layers[i].type == layerparam.LayerType.Value('RELU'):
                        self.components[netconfig.layers[i].bottom[0]].type = 'relu'
                    elif netconfig.layers[i].type == layerparam.LayerType.Value('DROPOUT'):
                        self.components[netconfig.layers[i].bottom[0]].droprate = netconfig.layers[i].dropout_param.dropout_ratio
                    else:
                        print 'Not Implemented Neuron'
                        print netconfig.layers[i].name
                        assert False
            else:
                assert netconfig.layers[i].type == layerparam.LayerType.Value('DATA')

        #Get the output
        for layername in self.components:
            if len(self.components[layername].ff_connect) == 0:
                self.outputcomp.append(layername)

        print "Input Component"
        for incomp in self.inputcomp:
            print incomp
        print "Output Component"
        for outcomp in self.outputcomp:
            print outcomp
        print "Connections"
        for conn in self.connections:
            print conn


    def ff(self, data):
        print "Network Feed Forward"
        #init status
        connstatus = dict()
        compstatus = dict()
        jobqueue = []

        for conn in self.connections:
            connstatus[conn] = False
        for comp in self.components:
            compstatus[comp] = False
        
        #mark the incomp ready
        for incomp in self.inputcomp:
            compstatus[incomp] = True
        
        #mark the ready
        for incomp in self.inputcomp:
            #check whether the incomp related connection is ready
            for incomp_conn in self.components[incomp].ff_connect:
                connstatus[incomp_conn] = True
                for conn_bottom in self.connections[incomp_conn].bottom:
                    if compstatus[conn_bottom] == False:
                        connstatus[incomp_conn] = False
                        break
                if connstatus[incomp_conn] == True:
                    jobqueue.insert(0, incomp_conn)

        while(len(jobqueue)!=0):
            curjob = jobqueue.pop()
            #execute current job
            self.connections[curjob].ff()
            #trigger the followers
            for finishedcomp in self.connections[curjob].top:
                compstatus[finishedcomp] = True
            for finishedcomp in self.connections[curjob].top:
                for next_conn in self.components[finishedcomp].ff_connect:
                    if connstatus[next_conn] == True:
                        continue
                    connstatus[next_conn] = True
                    for next_conn_bottom in self.connections[next_conn].bottom:
                        if compstatus[next_conn_bottom] == False:
                            connstatus[next_conn] = False
                            break
                    if connstatus[next_conn] == True:
                        jobqueue.insert(0, next_conn)


    def bp(self, error):
        print 'bp'
                    

if __name__ == '__main__':
    newconfig = CaffeModelConfig(configfile = './GoogLeNet/train_val.prototxt')
    #newconfig = CaffeModelConfig(configfile = '/home/minjie/caffe/caffe/models/bvlc_reference_caffenet/train_val.prototxt')
    model = MinervaModel(newconfig.netconfig)
    data = []
    model.ff(data)



            






