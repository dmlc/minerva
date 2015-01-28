import owl
import owl.elewise as ele
import owl.conv as co
import numpy as np
import Queue
from caffe import *

class ComputeUnit(object):
    def __init__(self, params = None):
        self.params = params
        self.bottom = []
        self.top = []
        self.act = []
        self.sen = []
    def __str__(self):
        return 'N/A unit'
    def ff(self):
        pass
    def bp(self):
        pass
    def update(self):
        pass

class WeightedComputeUnit(ComputeUnit):
    def __init__(self, params):
        super(WeightedComputeUnit, self).__init__(params)
        self.name = params.name
        self.params = params
        # weights and bias
        self.weight = None
        self.weightdelta = None
        self.weightgrad = None
        self.bias = None
        self.biasdelta = None
        self.biasgrad = None

class NeuronUnit(ComputeUnit):
    def __init__(self, params):
        super(NeuronUnit, self).__init__(params)
    def ff(self):
        self.act[0] = self.ff(self.bottom[0].act[0])
    def bp(self):
        self.sen[0] = self.bp(self.top[0].sen[0])
    def ff(self, x):
        pass
    def bp(self, y):
        pass
    def __str__(self):
        return 'N/A neuron'

class LinearUnit(NeuronUnit):
    def ff(self, x):
        return x
    def bp(self, y):
        return y
    def __str__(self):
        return 'linear'

class SigmoidUnit(NeuronUnit):
    def ff(self, x):
        return ele.sigm(x)
    def bp(self, y):
        return ele.sigm_back(y)
    def __str__(self):
        return 'sigmoid'

class ReluUnit(NeuronUnit):
    def ff(self, x):
        self.ff_x = x
        return ele.relu(x)
    def bp(self, y):
        return ele.relu_back(y, self.ff_x)
    def __str__(self):
        return 'relu'

class TanhUnit(NeuronUnit):
    def ff(self, x):
        return ele.tanh(x)
    def bp(self, y):
        return ele.tanh_back(y)
    def __str__(self):
        return 'tanh'

class PoolingUnit(NeuronUnit):
    def __init__(self, params):
        super(PoolingUnit, self).__init__(params)
        if params.pool == PoolingParameter.PoolMethod.Value('MAX'):
            pool_ty = co.pool_op.max
        elif params.pool == PoolingParameter.PoolMethod.Value('AVE'):
            pool_ty = co.pool_op.avg
        self.pooler = co.Pooler(params.kernel_size, params.kernel_size, params.stride, params.stride, params.pad, params.pad, pool_ty)
    def ff(self, x):
        self.ff_x = x
        self.ff_y = self.pooler.ff(x)
        return self.ff_y
    def bp(self, y):
        return self.pooler.bp(y, self.ff_y, self.ff_x)
    def __str__(self):
        return 'pooling'

class DropoutUnit(NeuronUnit):
    def __init__(self, params):
        super(DropoutUnit, self).__init__(params)
    def ff(self, x):
        self.dropmask = owl.randb(x, self.params.dropout_ratio)
        return ele.mult(x, self.dropmask)
    def bp(self, y):
        return ele.mult(y, self.dropmask)
    def __str__(self):
        return 'dropout'

class SoftmaxUnit(NeuronUnit):
    def __init__(self, params):
        super(SoftmaxUnit, self).__init__(params)
    def ff(self, x):
        self.ff_y = co.softmax(x, co.soft_op.instance)
        return self.ff_y
    def bp(self, y):
        return ff_y - y
    def __str__(self):
        return 'softmax'

# TODO
class LRNUnit(NeuronUnit):
    def __init__(self, params):
        super(LRNUnit, self).__init__(params)
    def ff(self, x):
        return x
    def bp(self, y):
        return y
    def __str__(self):
        return 'lrn'

# TODO
class ConcatUnit(ComputeUnit):
    def __init__(self, params):
        super(ConcatUnit, self).__init__(params)

class FullyConnection(WeightedComputeUnit):
    def __init__(self, params):
        super(FullyConnection, self).__init__(params)
        
    def ff(self):
        shp = self.bottom[0].act[0].shape
        if len(shp) > 2:
            a = self.bottom[0].act[0].reshape([np.prod(shp[0:-1]), shp[-1]])
        else:
            a = self.bottom[0].act[0]
        return self.weight * a + self.bias
    def bp(self):
        self.weightgrad = self.top[0].sen[0] * self.bottom[0].act[0].Trans()
        self.biasgrad = self.top[0].sen[0].sum(0)
        s = self.weight.Trans() * self.top[0].sen[0]
        shp = self.bottom[0].act[0].shape
        if len(shp) > 2:
            s = s.reshape(shp)
        return s
    def __str__(self):
        return 'fc'

class ConvConnection(WeightedComputeUnit):
    def __init__(self, params):
        super(ConvConnection, self).__init__(params)
        self.conv_params = params.convolution_param
        self.convolver = co.Convolver(self.conv_params.pad, 
                self.conv_params.pad, self.conv_params.stride, self.conv_params.stride)
    def ff(self):
        return self.convolver.ff(self.bottom[0].act[0], self.weight, self.bias)
    def bp(self):
        self.weightgrad = self.convolver.weight_grad(self.top[0].sen[0], self.bottom[0].act[0])
        self.biasgrad = self.convolver.bias_grad(self.top[0].sen[0])
        return self.convolver.bp(self.top[0].sen[0], self.weight)
    def __str__(self):
        return 'conv'

class DataUnit(ComputeUnit):
    def __init__(self, params):
        super(DataUnit, self).__init__(params)
    def ff(self):
        pass
    def bp(self):
        pass
    def __str__(self):
        return 'data'

class LRNConnection(Connection):
    def __init__(self, name, local_size, alpha, beta):
        super(LRNConnection, self).__init__(name)
        self.lrner = co.Lrner(local_size, alpha, beta)
        self.scale = None
    def ff(self):
        self.scale = owl.zeros(self.bottom[0].get_act().shape)
        return self.lrner.ff(self.bottom[0].get_act(), self.scale)
    def bp(self):
        self.bottom[0].sen = self.lrner.bp(self.bottom[0].get_act(), self.top[0].get_act(), self.scale, self.top[0].sen)


class ConcatConnection(Connection):
    def __init__(self, name, concat_dim):
        super(ConcatConnection, self).__init__(name)
        self.concat_dim = concat_dim
    def ff(self):
        assert len(self.bottom) > 1
        narrays = []
        for i in range(len(self.bottom)):
            narrays.append(self.bottom[i].get_act())
        return owl.concat(narrays, self.concat_dim)

    def bp(self):
        assert len(self.bottom) > 1
        st_off = 0
        for i in range(len(self.bottom)):
            slice_count = self.bottom[i].get_act().shape[self.concat_dim]
            self.bottom[i].sen = owl.slice(self.top[0].sen, self.concat_dim, st_off, slice_count)
            st_off += slice_count
        pass

class Net:
    def __init__(self):
        self.units = {}
        self.adjacent = {}
        self.reverse_adjacent = {}

    def add_unit(self, name, unit):
        self.units[name] = unit
        self.adjacent[name] = []
        self.reverse_adjacent[name] = []

    def has_unit(self, name):
        return name in self.units

    def connect(self, n1, n2):
        self.adjacent[n1].append(n2)
        self.reverse_adjacent[n2].append(n1)
        l1 = self.units[n1]
        l2 = self.units[n2]
        l1.top.append(l2)
        l2.bottom.append(l1)

    def _toporder(self):
        depcount = {unit : len(inunits) for unit, inunits in self.reverse_adjacent.iteritems()}
        queue = Queue.Queue()
        for unit, count in depcount.iteritems():
            if count == 0:
                queue.put(unit)
        while not queue.empty():
            unit = queue.get()
            yield self.units[unit]
            for l in self.adjacent[unit]:
                depcount[l] -= 1
                if depcount[l] == 0:
                    queue.put(l)

    def _reverse_toporder(self):
        depcount = {unit : len(outunits) for unit, outunits in self.adjacent.iteritems()}
        queue = Queue.Queue()
        for unit, count in depcount.iteritems():
            if count == 0:
                queue.put(unit)
        while not queue.empty():
            unit = queue.get()
            yield self.units[unit]
            for l in self.reverse_adjacent[unit]:
                depcount[l] -= 1
                if depcount[l] == 0:
                    queue.put(l)

    def ff(self):
        for l in self._toporder():
            l.ff()

    def bp(self):
        for l in self._reverse_toporder():
            l.bp()
    
    def __str__(self):
        ret = 'digraph G {\n'
        for k, l in self.adjacent.iteritems():
            for n in l:
                ret += '"' + str(k) + '"->"' + str(n) + '"\n'
        return ret + '}\n'

class TestUnit(ComputeUnit):
    def __init__(self, name):
        super(TestUnit, self).__init__()
        self.name = name
    def ff(self):
        print 'ff:', self.name
    def bp(self):
        print 'bp:', self.name

if __name__ == '__main__':
    net = Net()
    net.add_unit('l1', TestUnit('l1'))
    net.add_unit('l2', TestUnit('l2'))
    net.add_unit('l3', TestUnit('l3'))
    net.add_unit('l4', TestUnit('l4'))
    net.add_unit('l5', TestUnit('l5'))
    net.add_unit('l6', TestUnit('l6'))
    net.add_unit('l7', TestUnit('l7'))
    net.add_unit('l8', TestUnit('l8'))
    net.connect('l1', 'l2')
    net.connect('l1', 'l3')
    net.connect('l2', 'l4')
    net.connect('l3', 'l5')
    net.connect('l4', 'l5')
    net.connect('l5', 'l6')
    net.connect('l5', 'l7')
    net.connect('l7', 'l8')
    net.ff()
    net.bp()
