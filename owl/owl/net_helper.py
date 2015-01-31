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
        stacked_layers = {}
        rev_stacked_layers = {}
        top_name_to_layer = {}
        # 1. record name and its caffe.LayerParameter data in a map
        # 2. some layers is stacked into one in caffe's configure format
        for l in self.netconfig.layers:
            owl_struct = self._convert_type(l)
            if owl_struct != None:
                uid = owl_net.add_unit(owl_struct)
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
            return net.PoolingUnit(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('RELU'):
            return net.ReluUnit(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('SIGMOID'):
            return net.SigmoidUnit(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('SOFTMAX_LOSS'):
            return net.SoftmaxUnit(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('TANH'):
            return net.TanhUnit(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('DROPOUT'):
            return net.DropoutUnit(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('LRN'):
            return net.LRNUnit(caffe_layer)
        elif ty == LayerParameter.LayerType.Value('CONCAT'):
            return net.ConcatUnit(caffe_layer)
        else:
            print "Not implemented type:", LayerParameter.LayerType.Name(caffe_layer.type)
            return None

if __name__ == "__main__":
    builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
    owl_net = net.Net()
    builder.build_net(owl_net)
