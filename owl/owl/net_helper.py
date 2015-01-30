import sys
import owl.net as net
from caffe import *
from google.protobuf import text_format

class INetBuilder:
    def build_net(self, owl_net):
        pass

class CaffeNetBuilder:
    def __init__(self, net_file, solver_file):
        print 'Caffe network file:', net_file
        print 'Caffe solver file:', solver_file
        with open(net_file, 'r') as f:
            self.netconfig = NetParameter()
            text_format.Merge(str(f.read()), self.netconfig)
        with open(solver_file, 'r') as f:
            self.solverconfig = SolverParameter()
            text_format.Merge(str(f.read()), self.solverconfig)

    def build_net(self, owl_net):
        owl_net = net.Net()
        caffe_layers = {}
        caffe_to_owl_map = {}
        stacked_layers = {}
        rev_stacked_layers = {}
        top_name_to_layer = {}
        # 1. record name and its caffe.LayerParameter data in a map
        # 2. some layers is stacked into one in caffe's configure format
        for l in self.netconfig.layers:
            caffe_layers[l.name] = l
            caffe_to_owl_map[l.name] = self._convert_type(l)
            stacked_layers[l.name] = [l.name]
            rev_stacked_layers[l.name] = l.name
            if len(l.bottom) == 1 and len(l.top) == 1 and l.bottom[0] == l.top[0]:
                stack_to = l.bottom[0]
                stacked_layers[stack_to].append(l.name)
                rev_stacked_layers[l.name] = stack_to
            else:
                for t in l.top:
                    top_name_to_layer[t] = l.name
            owl_net.add_unit(l.name, caffe_to_owl_map[l.name])
        # 3. connect
        for base_layer, stacks in stacked_layers.iteritems():
            if caffe_to_owl_map[base_layer] != None and rev_stacked_layers[base_layer] == base_layer:
                i1 = 0
                i2 = i1 + 1
                while i2 < len(stacks):
                    owl_net.connect(stacks[i1], stacks[i2])
                    i1 = i1 + 1
                    i2 = i2 + 1
                for btm in caffe_layers[base_layer].bottom:
                    if btm == 'label': #hack
                        btm = 'data'
                    real_btm = stacked_layers[rev_stacked_layers[btm]][-1]
                    owl_net.connect(real_btm, base_layer)
        print owl_net

    def _convert_type(self, caffe_layer):
        ty = caffe_layer.type
        if ty == LayerParameter.LayerType.Value('DATA'):
            return net.DataUnit(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('INNER_PRODUCT'):
            return net.FullyConnection(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('CONVOLUTION'):
            return net.ConvConnection(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('POOLING'):
            return net.PoolingUnit(caffe_layer.pooling_param)
        elif ty == LayerParameter.LayerType.Value('RELU'):
            return net.ReluUnit(caffe_layer.relu_param)
        elif ty == LayerParameter.LayerType.Value('SIGMOID'):
            return net.SigmoidUnit(caffe_layer.sigmoid_param)
        elif ty == LayerParameter.LayerType.Value('SOFTMAX_LOSS'):
            return net.SoftmaxUnit(caffe_layer.softmax_param)
        elif ty == LayerParameter.LayerType.Value('TANH'):
            return net.TanhUnit(caffe_layer.tanh_param)
        elif ty == LayerParameter.LayerType.Value('DROPOUT'):
            return net.DropoutUnit(caffe_layer.dropout_param)
        elif ty == LayerParameter.LayerType.Value('LRN'):
            return net.LRNUnit(caffe_layer.lrn_param)
        elif ty == LayerParameter.LayerType.Value('CONCAT'):
            return net.ConcatUnit(caffe_layer.concat_param)
        else:
            print "Not implemented type:", LayerParameter.LayerType.Name(caffe_layer.type)
            return None

if __name__ == "__main__":
    builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
    owl_net = net.Net()
    builder.build_net(owl_net)
