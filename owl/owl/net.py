import owl
import owl.elewise as ele
import owl.conv as co
import numpy as np
import Queue
from caffe import *

class ComputeUnit(object):
    def __init__(self, params):
        self.params = params
        self.name = params.name
        self.btm_names = []
        self.top_names = []
    def __str__(self):
        return 'N/A unit'
    def forward(self, from_btm, to_top):
        pass
    def backward(self, from_top, to_btm):
        pass
    def update(self):
        pass

class ComputeUnitSimple(ComputeUnit):
    def __init__(self, params):
        super(ComputeUnitSimple, self).__init__(params)
    def forward(self, from_btm, to_top):
        to_top[self.top_names[0]] = self.ff(from_btm[self.btm_names[0]])
    def ff(self, act):
        pass
    def backward(self, from_top, to_btm):
        to_btm[self.btm_names[0]] = self.bp(from_top[self.top_names[0]])
        pass
    def bp(self, sen):
        pass

class WeightedComputeUnit(ComputeUnitSimple):
    def __init__(self, params):
        super(WeightedComputeUnit, self).__init__(params)
        self.params = params
        # weights and bias
        self.weight = None
        self.weightdelta = None
        self.weightgrad = None
        self.bias = None
        self.biasdelta = None
        self.biasgrad = None

class LinearUnit(ComputeUnitSimple):
    def ff(self, x):
        return x
    def bp(self, y):
        return y
    def __str__(self):
        return 'linear'

class SigmoidUnit(ComputeUnitSimple):
    def ff(self, x):
        return ele.sigm(x)
    def bp(self, y):
        return ele.sigm_back(y)
    def __str__(self):
        return 'sigmoid'

class ReluUnit(ComputeUnitSimple):
    def ff(self, x):
        self.ff_x = x
        return ele.relu(x)
    def bp(self, y):
        return ele.relu_back(y, self.ff_x)
    def __str__(self):
        return 'relu'

class TanhUnit(ComputeUnitSimple):
    def ff(self, x):
        return ele.tanh(x)
    def bp(self, y):
        return ele.tanh_back(y)
    def __str__(self):
        return 'tanh'

class PoolingUnit(ComputeUnitSimple):
    def __init__(self, params):
        super(PoolingUnit, self).__init__(params)
        ppa = params.pooling_param
        if ppa.pool == PoolingParameter.PoolMethod.Value('MAX'):
            pool_ty = co.pool_op.max
        elif ppa.pool == PoolingParameter.PoolMethod.Value('AVE'):
            pool_ty = co.pool_op.avg
        self.pooler = co.Pooler(ppa.kernel_size, ppa.kernel_size, ppa.stride, ppa.stride, ppa.pad, ppa.pad, pool_ty)
    def ff(self, x):
        self.ff_x = x
        self.ff_y = self.pooler.ff(x)
        return self.ff_y
    def bp(self, y):
        return self.pooler.bp(y, self.ff_y, self.ff_x)
    def __str__(self):
        return 'pooling'

class DropoutUnit(ComputeUnitSimple):
    def __init__(self, params):
        super(DropoutUnit, self).__init__(params)
    def ff(self, x):
        self.dropmask = owl.randb(x, self.params.dropout_param.dropout_ratio)
        return ele.mult(x, self.dropmask)
    def bp(self, y):
        return ele.mult(y, self.dropmask)
    def __str__(self):
        return 'dropout'

class SoftmaxUnit(ComputeUnitSimple):
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
class LRNUnit(ComputeUnitSimple):
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
    def ff(self, from_btm, to_top):
        pass
    def bp(self, from_top, to_btm):
        pass
    def __str__(self):
        return 'concat'
'''
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
'''

class FullyConnection(WeightedComputeUnit):
    def __init__(self, params):
        super(FullyConnection, self).__init__(params)
        
    def ff(self, act):
        shp = act.shape
        if len(shp) > 2:
            a = act.reshape([np.prod(shp[0:-1]), shp[-1]])
        else:
            a = act
        self.ff_act = act # save ff value
        return self.weight * a + self.bias
    def bp(self, sen):
        self.weightgrad = sen * self.ff_act.Trans()
        self.biasgrad = act.sum(0)
        s = self.weight.Trans() * sen 
        shp = self.ff_act.shape
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
    def ff(self, act):
        self.ff_act = act
        return self.convolver.ff(act, self.weight, self.bias)
    def bp(self, sen):
        self.weightgrad = self.convolver.weight_grad(sen, self.weight, self.ff_act)
        self.biasgrad = self.convolver.bias_grad(sen)
        return self.convolver.bp(sen, self.ff_act, self.weight)
    def __str__(self):
        return 'conv'

# TODO
class DataUnit(ComputeUnit):
    def __init__(self, params):
        super(DataUnit, self).__init__(params)
    def forward(self, from_btm, to_top):
        pass
    def backward(self, from_top, to_btm):
        pass
    def __str__(self):
        return 'data'

class Net:
    def __init__(self):
        self.units = []
        self.adjacent = []
        self.reverse_adjacent = []

    def add_unit(self, unit):
        uid = len(self.units)
        self.units.append(unit)
        self.adjacent.append([])
        self.reverse_adjacent.append([])
        return uid

    def connect(self, u1, u2):
        self.adjacent[u1].append(u2)
        self.reverse_adjacent[u2].append(u1)

    def _toporder(self):
        depcount = [len(inunits) for inunits in self.reverse_adjacent]
        queue = Queue.Queue()
        for unit in range(len(depcount)):
            count = depcount[unit]
            if count == 0:
                queue.put(unit)
        while not queue.empty():
            unit = queue.get()
            yield unit
            for l in self.adjacent[unit]:
                depcount[l] -= 1
                if depcount[l] == 0:
                    queue.put(l)

    def _reverse_toporder(self):
        depcount = [len(outunits) for outunits in self.adjacent]
        queue = Queue.Queue()
        for unit in range(len(depcount)):
            count = depcount[unit]
            if count == 0:
                queue.put(unit)
        while not queue.empty():
            unit = queue.get()
            yield unit
            for l in self.reverse_adjacent[unit]:
                depcount[l] -= 1
                if depcount[l] == 0:
                    queue.put(l)

    def forward(self):
        unit_to_tops = [{} for name in self.units]
        for u in self._toporder():
            from_btm = {}
            for btm in self.reverse_adjacent[u]:
                from_btm.update(unit_to_tops[btm])
            self.units[u].forward(from_btm, unit_to_tops[u])

    def backward(self):
        unit_to_btms = [{} for name in self.units]
        for u in self._reverse_toporder():
            from_top = {}
            for top in self.adjacent[u]:
                from_top.update(unit_to_btms[top])
            self.units[u].backward(from_top, unit_to_btms[u])
    
    def __str__(self):
        ret = 'digraph G {\n'
        for uid in range(len(self.units)):
            ret += 'n' + str(uid) + ' [label="' + self.units[uid].name + '"]\n'
        for uid in range(len(self.units)):
            for nuid in self.adjacent[uid]:
                ret += 'n' + str(uid) + ' -> n' + str(nuid) + '\n'
        return ret + '}\n'


############### Test code
class _StartUnit(ComputeUnit):
    def __init__(self, name):
        self.name = name
        self.btm_names = []
        self.top_names = []
    def forward(self, from_btm, to_top):
        print 'ff|start name:', self.name
        to_top[self.top_names[0]] = 0
    def backward(self, from_top, to_btm):
        pass

class _EndUnit(ComputeUnit):
    def __init__(self, name):
        self.name = name
        self.btm_names = []
        self.top_names = []
    def forward(self, from_btm, to_top):
        pass
    def backward(self, from_top, to_btm):
        print 'bp|end name:', self.name
        to_btm[self.btm_names[0]] = 0

class _TestUnit(ComputeUnitSimple):
    def __init__(self, name):
        self.name = name
        self.btm_names = []
        self.top_names = []
    def ff(self, x):
        print 'ff|name:', self.name, 'val:', x
        return x + 1
    def bp(self, y):
        print 'bp|name:', self.name, 'val:', y
        return y - 1

if __name__ == '__main__':
    net = Net()
    us = _StartUnit('s')
    u1 = _TestUnit('u1')
    u2 = _TestUnit('u2')
    u3 = _TestUnit('u3')
    u4 = _TestUnit('u4')
    u5 = _TestUnit('u5')
    ue = _EndUnit('e')
    us.top_names = ['s']
    u1.btm_names = ['s']
    u1.top_names = ['u1']
    u2.btm_names = ['u1']
    u2.top_names = ['u2']
    u3.btm_names = ['u2']
    u3.top_names = ['u3']
    u4.btm_names = ['u3']
    u4.top_names = ['u4']
    u5.btm_names = ['u4']
    u5.top_names = ['u5']
    ue.btm_names = ['u5']
    ls = net.add_unit(us)
    l1 = net.add_unit(u1)
    l2 = net.add_unit(u2)
    l3 = net.add_unit(u3)
    l4 = net.add_unit(u4)
    l5 = net.add_unit(u5)
    le = net.add_unit(ue)
    net.connect(ls, l1)
    net.connect(l1, l2)
    net.connect(l2, l3)
    net.connect(l3, l4)
    net.connect(l4, l5)
    net.connect(l5, le)
    net.forward()
    net.backward()
