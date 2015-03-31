import os
import sys
import owl.net as net
from caffe import *
from google.protobuf import text_format
import numpy as np
import owl
from PIL import Image
import subprocess

class INetBuilder:
    def build_net(self, owl_net):
        pass

class CaffeNetBuilder:
    def __init__(self, solver_file):
        print 'Caffe solver file:', solver_file
        with open(solver_file, 'r') as f:
            self.solverconfig = SolverParameter()
            text_format.Merge(str(f.read()), self.solverconfig)
        self.net_file = self.solverconfig.net
        print 'Caffe network file:', self.net_file
        with open(self.net_file, 'r') as f:
            self.netconfig = NetParameter()
            text_format.Merge(str(f.read()), self.netconfig)
        self.snapshot_dir = self.solverconfig.snapshot_prefix
        print 'Snapshot Dir: %s' % (self.snapshot_dir)
    
    def build_net(self, owl_net, num_gpu):
        #set globle lr and wd
        owl_net.base_lr = self.solverconfig.base_lr
        owl_net.current_lr = self.solverconfig.base_lr
        owl_net.base_weight_decay = self.solverconfig.weight_decay
        owl_net.momentum = self.solverconfig.momentum
        owl_net.solver = self.solverconfig
        owl_net.lr_policy = self.solverconfig.lr_policy

        stacked_layers = {}
        rev_stacked_layers = {}
        top_name_to_layer = {}
        # 1. record name and its caffe.V1LayerParameter data in a map
        # 2. some layers is stacked into one in caffe's configure format
        for l in self.netconfig.layers:
            owl_struct = self._convert_type(l, num_gpu)
            
            if owl_struct != None:
                uid = owl_net.add_unit(owl_struct)
                
                #handle IO, may need better         
                ty = l.type
                if ty == V1LayerParameter.LayerType.Value('DATA'):
                    if len(l.include) != 0 and l.include[0].phase == Phase.Value('TRAIN'):
                        owl_net.batch_size = l.data_param.batch_size
                elif ty == V1LayerParameter.LayerType.Value('IMAGE_DATA'):
                    if len(l.include) != 0 and l.include[0].phase == Phase.Value('TRAIN'):
                        owl_net.batch_size = l.image_data_param.batch_size
                elif ty == V1LayerParameter.LayerType.Value('WINDOW_DATA'):
                    if len(l.include) != 0 and l.include[0].phase == Phase.Value('TRAIN'):
                        owl_net.batch_size = l.window_data_param.batch_size
                elif ty == V1LayerParameter.LayerType.Value('SOFTMAX_LOSS'):
                    owl_net.loss_uids.append(uid)
                elif ty == V1LayerParameter.LayerType.Value('ACCURACY'):
                    owl_net.accuracy_uids.append(uid)
                
                # stack issues
                #stacked_layers[l.name] = [uid]
                #rev_stacked_layers[uid] = l.name
                if len(l.bottom) == 1 and len(l.top) == 1 and l.bottom[0] == l.top[0]:
                    # top name
                    top_name_to_layer[l.name] = [uid]
                    owl_net.units[uid].top_names = [l.name]

                    stack_to = l.bottom[0]
                    if not stack_to in stacked_layers:
                        stacked_layers[stack_to] = [top_name_to_layer[stack_to][0]]

                    # bottom name
                    btm_uid = stacked_layers[stack_to][-1]
                    owl_net.units[uid].btm_names = [owl_net.units[btm_uid].top_names[0]]

                    stacked_layers[stack_to].append(uid)
                    rev_stacked_layers[uid] = stack_to
                else:
                    # top name
                    for top in l.top:
                        if not top in top_name_to_layer:
                            top_name_to_layer[top] = []
                        top_name_to_layer[top].append(uid)
                    owl_net.units[uid].top_names = list(l.top)
                    # bottom name
                    btm_names = []
                    for btm in l.bottom:
                        if btm in stacked_layers:
                            btm_names.append(owl_net.units[stacked_layers[btm][-1]].top_names[0])
                        else:
                            btm_names.append(btm)
                    owl_net.units[uid].btm_names = btm_names

        # 3. connect
        for uid in range(len(owl_net.units)):
            for btm in owl_net.units[uid].btm_names:
                for btm_uid in top_name_to_layer[btm]:
                    owl_net.connect(btm_uid, uid)
        #print owl_net

    def _convert_type(self, caffe_layer, num_gpu):
        ty = caffe_layer.type
        if ty == V1LayerParameter.LayerType.Value('DATA'):
            return net.LMDBDataUnit(caffe_layer, num_gpu)
        elif ty == V1LayerParameter.LayerType.Value('IMAGE_DATA'):
            return net.ImageDataUnit(caffe_layer, num_gpu)
        elif ty == V1LayerParameter.LayerType.Value('WINDOW_DATA'):
            return net.ImageWindowDataUnit(caffe_layer, num_gpu)
        elif ty == V1LayerParameter.LayerType.Value('INNER_PRODUCT'):
            return net.FullyConnection(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('CONVOLUTION'):
            return net.ConvConnection(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('POOLING'):
            return net.PoolingUnit(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('RELU'):
            return net.ReluUnit(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('SIGMOID'):
            return net.SigmoidUnit(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('SOFTMAX_LOSS'):
            return net.SoftmaxUnit(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('TANH'):
            return net.TanhUnit(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('DROPOUT'):
            return net.DropoutUnit(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('LRN'):
            return net.LRNUnit(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('CONCAT'):
            return net.ConcatUnit(caffe_layer)
        elif ty == V1LayerParameter.LayerType.Value('ACCURACY'):
            return net.AccuracyUnit(caffe_layer)
        else:
            print "Not implemented type:", V1LayerParameter.LayerType.Name(caffe_layer.type)
            return None
    
    def init_net_from_file(self, owl_net, weightpath, epochidx):
        weightpath = "%ssnapshot%d/" % (weightpath, epochidx)
        for i in range(len(owl_net.units)):
            if isinstance(owl_net.units[i], net.FullyConnection):
                #print owl_net.units[i].name
                layername = owl_net.units[i].name
                layername = layername.replace("/","_")
                weightname = '%s%s_weights.dat' % (weightpath, layername)
                
                wshape = owl_net.units[i].wshape
                if os.path.isfile(weightname):
                    npweight = np.fromfile(weightname, dtype = np.float32)
                    length = np.shape(npweight)[0]
                    if length == owl_net.units[i].in_shape[0] * owl_net.units[i].out_shape[0]:
                        owl_net.units[i].weight = owl.from_numpy(npweight).reshape(wshape)
                        weightname = '%s%s_weightdelta.dat' % (weightpath, layername)
                        if os.path.isfile(weightname):
                            npweightdelta = np.fromfile(weightname, dtype = np.float32)
                            owl_net.units[i].weightdelta = owl.from_numpy(npweightdelta).reshape(wshape) 
                    else:
                        print "Weight Need Reinit %s" % (owl_net.units[i].name)
                else:
                    print "Weight Need Reinit %s" % (owl_net.units[i].name)
            
                biasname = '%s%s_bias.dat' % (weightpath, layername)
                bshape = owl_net.units[i].bshape
                if os.path.isfile(biasname):
                    npbias = np.fromfile(biasname, dtype = np.float32)
                    length = np.shape(npbias)[0]
                    if length == owl_net.units[i].out_shape[0]:
                        owl_net.units[i].bias = owl.from_numpy(npbias).reshape(bshape)
                        biasname = '%s%s_biasdelta.dat' % (weightpath, layername)
                        if os.path.isfile(biasname):
                            npbiasdetla = np.fromfile(biasname, dtype = np.float32)
                            owl_net.units[i].biasdelta = owl.from_numpy(npbiasdetla).reshape(bshape)
                    else:
                        print "Bias Need Reinit %s" % (owl_net.units[i].name)
                
            if isinstance(owl_net.units[i], net.ConvConnection):
                #print owl_net.units[i].name
                layername = owl_net.units[i].name
                layername = layername.replace("/","_")
                conv_params = owl_net.units[i].conv_params
                
                weightname = '%s%s_weights.dat' % (weightpath, layername)
                wshape = owl_net.units[i].wshape
                if os.path.isfile(weightname):
                    npweight = np.fromfile(weightname, dtype = np.float32)
                    length = np.shape(npweight)[0]
                    if length == owl_net.units[i].in_shape[2] * owl_net.units[i].out_shape[2] * conv_params.kernel_size * conv_params.kernel_size:
                        owl_net.units[i].weight = owl.from_numpy(npweight).reshape(wshape)
                        weightname = '%s%s_weightdelta.dat' % (weightpath, layername)
                        if os.path.isfile(weightname):
                            npweightdelta = np.fromfile(weightname, dtype = np.float32)
                            owl_net.units[i].weightdelta = owl.from_numpy(npweightdelta).reshape(wshape)
                    else:
                        print "Conv Weight Need Reinit %s" % (owl_net.units[i].name)
                else:
                    print "Conv Weight Need Reinit %s" % (owl_net.units[i].name)
              
                biasname = '%s%s_bias.dat' % (weightpath, layername)
                bshape = owl_net.units[i].bshape
                if os.path.isfile(biasname):
                    npbias = np.fromfile(biasname, dtype = np.float32)
                    length = np.shape(npbias)[0]
                    if length == owl_net.units[i].out_shape[2]:
                        owl_net.units[i].bias = owl.from_numpy(npbias).reshape(bshape)
                        biasname = '%s%s_biasdelta.dat' % (weightpath, layername)
                        if os.path.isfile(biasname):
                            npbiasdetla = np.fromfile(biasname, dtype = np.float32)
                            owl_net.units[i].biasdelta = owl.from_numpy(npbiasdetla).reshape(bshape)
                    else:
                        print "Conv Bias Need Reinit %s" % (owl_net.units[i].name)
                else:
                    print "Conv Bias Need Reinit %s" % (owl_net.units[i].name)

    
    def save_net_to_file(self, owl_net, weightpath, epochidx):
        weightpath = "%ssnapshot%d/" % (weightpath, epochidx)
        cmd = "mkdir %s" % (weightpath)
        res = subprocess.call(cmd, shell=True)
        for i in range(len(owl_net.units)):
            if isinstance(owl_net.units[i], net.ConvConnection) or isinstance(owl_net.units[i], net.FullyConnection):
                #print owl_net.units[i].name
                layername = owl_net.units[i].name
                layername = layername.replace("/","_")
                weightname = '%s%s_weights.dat' % (weightpath, layername)
                wshape = owl_net.units[i].weight.shape 
                length = np.prod(wshape)
                npweight = owl_net.units[i].weight.to_numpy().reshape(length)
                npweight.tofile(weightname)

                weightname = '%s%s_weightdelta.dat' % (weightpath, layername)
                npweightdelta = owl_net.units[i].weightdelta.to_numpy().reshape(length)
                npweightdelta.tofile(weightname)
                
                biasname = '%s%s_bias.dat' % (weightpath, layername)
                bshape = owl_net.units[i].bias.shape
                length = np.prod(bshape)
                npbias = owl_net.units[i].bias.to_numpy().reshape(length)
                npbias.tofile(biasname)

                biasname = '%s%s_biasdelta.dat' % (weightpath, layername)
                npbiasdetla = owl_net.units[i].biasdelta.to_numpy().reshape(length)
                npbiasdetla.tofile(biasname)

class CaffeModelLoader:
    def __init__(self, model_file, status_file, weightdir, snapshot):
        netparam = NetParameter()
        layerparam = V1LayerParameter()
        with open(model_file, 'rb') as f:
            netparam.ParseFromString(f.read())
       
        '''
        netdelta = SolverState()
        with open(status_file, 'rb') as fd:
            netdelta.ParseFromString(fd.read())
        '''

        cmd = 'mkdir %s' % (weightdir) 
        res = subprocess.call(cmd, shell=True)

        cmd = 'mkdir %s/snapshot%d' % (weightdir, snapshot) 
        res = subprocess.call(cmd, shell=True)

        print len(netparam.layers)
        curweights = 0
        for i in range(len(netparam.layers)):
            #print '%d %d' % (i, curweights)
            if hasattr(netparam.layers[i], 'blobs') and len(netparam.layers[i].blobs) == 2:
                layername = netparam.layers[i].name
                layername = layername.replace("/","_")
                filename = '%s/snapshot%d/%s_weights.dat' % (weightdir, snapshot, layername)
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
                
                filename = '%s/snapshot%d/%s_bias.dat' % (weightdir, snapshot, layername)
                np.array(netparam.layers[i].blobs[1].data, dtype=np.float32).tofile(filename)
              
if __name__ == "__main__":
    CaffeModelLoader('/home/tianjun/caffe/caffe/models/bvlc_googlenet/bvlc_googlenet_quick_iter_40.caffemodel', '/home/tianjun/caffe/caffe/models/bvlc_googlenet/bvlc_googlenet_quick_iter_40.solverstate', '/home/tianjun/models/GoogModel/', 0)
    
    #CaffeModelLoader('/home/tianjun/caffe/caffe/models/bvlc_alexnet/caffe_alexnet_train_iter_20.caffemodel', '/home/tianjun/caffe/caffe/models/bvlc_alexnet/Minervamodel/')
    
    '''
    builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
    owl_net = net.Net()
    builder.build_net(owl_net)
    owl_net.forward()
    '''
