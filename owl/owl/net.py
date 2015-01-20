import owl
import owl.elewise as ele
import owl.conv as co
import numpy as np

class Neuron(object):
    def __init__(self):
        pass

    def ff(self, x):
        pass

    def bp(self, y, ff_y, ff_x):
        pass

class LinearNeuron(Neuron):
    def ff(self, x):
        return x
    def bp(self, y, ff_y, ff_x):
        return y

class SigmoidNeuron(Neuron):
    def ff(self, x):
        return ele.sigm(x)
    def bp(self, y, ff_y, ff_x):
        return ele.sigm_back(y)

class ReluNeuron(Neuron):
    def ff(self, x):
        return ele.relu(x)
    def bp(self, y, ff_y, ff_x):
        return ele.relu_back(y, ff_x)

class TahnNeuron(Neuron):
    def ff(self, x):
        return ele.tahn(x)
    def bp(self, y, ff_y, ff_x):
        return ele.tahn_back(y)

class Layer(object):
    def __init__(self, name, neuron, dropout):
        self.name = name
        self.neuron = neuron
        self.dropout = dropout
        self.pre_nonlinear = None
        self.act = None
        self.sen = None
        self.dropmask = None
        
        # layer dimension
        self.dim = []

        # connectivity
        self.ff_conns = []
        self.bp_conns = []

    def get_act(self):
        if type(self.neuron) == LinearNeuron:
            return self.pre_nonlinear
        else:
            return self.act

    def ff(self):
        #some of the pre_nonlinear doesn't need to be stored
        if type(self.neuron) == LinearNeuron or type(self.neuron) == SigmoidNeuron or type(self.neuron) == TahnNeuron:
            self.pre_nonlinear = self.neuron.ff(self.pre_nonlinear)
            if self.dropout != 0.0:
                self.dropmask = owl.randb(self.pre_nonlinear.shape, self.dropout)
                self.pre_nonlinear = ele.mult(self.pre_nonlinear, self.dropmask)
            return self.pre_nonlinear
        else:
            self.act = self.neuron.ff(self.pre_nonlinear)
            if self.dropout != 0.0:
                self.dropmask = owl.randb(self.act.shape, self.dropout)
                self.act = ele.mult(self.act, self.dropmask)
            return self.act

    def bp(self):
        if self.dropout != 0.0:
            self.sen = ele.mult(self.sen, self.dropmask)
            self.dropmask = None
        self.sen = self.neuron.bp(self.sen, self.act, self.pre_nonlinear)
        #self.pre_nonlinear = None

class Connection(object):
    def __init__(self, name):
        self.name = name
        self.bottom = []
        self.top = []

    def weightinit(self, weightshape, initer):
        #TODO: hack, not all initer implemented, just the load from file
        return owl.from_numpy(np.fromfile(initer, dtype=np.float32)).reshape(weightshape)

    def ff(self):
        pass

    def bp(self):
        pass

    def update(nsamples, mom, lr, wd):
        pass

class FullyConnection(Connection):
    def __init__(self, name, wshape, bshape, winiter, biniter):
        super(FullyConnection, self).__init__(name)
        self.weight = self.weightinit(wshape, winiter)
        self.weightdelta = owl.zeros(wshape)
        self.weightgrad = None
        self.bias = self.weightinit(bshape, biniter)
        self.biasdelta = owl.zeros(bshape)
        self.biasgrad = None
        # learning rate of this connection
        self.blobs_lr = [0, 0]
        self.blobs_wd = [0, 0]
        
    def ff(self):
        assert len(self.bottom) == 1
        shp = self.bottom[0].get_act().shape
        if len(shp) > 2:
            a = self.bottom[0].get_act().reshape([np.prod(shp[0:-1]), shp[-1]])
        else:
            a = self.bottom[0].get_act()
        return self.weight * a + self.bias
    def bp(self):
        assert len(self.top) == 1 and len(self.bottom) == 1
        shp = self.bottom[0].get_act().shape
        if len(shp) > 2:
            botact = self.bottom[0].get_act().reshape([np.prod(shp[0:-1]), shp[-1]])
        else:
            botact = self.bottom[0].get_act()
        self.weightgrad = self.top[0].sen * botact.trans()
        self.biasgrad = self.top[0].sen.sum(0)
        s = self.weight.trans() * self.top[0].sen
        shp = self.bottom[0].get_act().shape
        if len(shp) > 2:
            s = s.reshape(shp)
        self.bottom[0].sen = s

    def update(self, nsamples, mom, lr, wd):
        self.weightdelta = mom * self.weightdelta - lr * self.blobs_lr[0] / nsamples * self.weightgrad - lr * self.blobs_lr[0] * self.blobs_wd[0] * self.weight
        self.weight += self.weightdelta
        self.weightgrad = None # reset the grad to None to free the space
        self.biasdelta = mom * self.biasdelta - lr * self.blobs_lr[1] / nsamples * self.biasgrad - lr * self.blobs_lr[1] * self.blobs_wd[1] * self.bias
        self.bias += self.biasdelta
        self.biasgrad = None # reset the grad to None to free the space

class ConvConnection(Connection):
    def __init__(self, name, wshape, bshape, winiter, biniter, pad, stride):
        super(ConvConnection, self).__init__(name)
        self.weight = self.weightinit(wshape, winiter)
        self.weightdelta = owl.zeros(wshape)
        self.weightgrad = None
        self.bias = self.weightinit(bshape, biniter)
        self.biasdelta = owl.zeros(bshape)
        self.biasgrad = None
        # learning rate of this connection
        self.blobs_lr = [0, 0]
        self.blobs_wd = [0, 0]
        self.convolver = co.Convolver(pad, pad, stride, stride)
    def ff(self):
        assert len(self.bottom) == 1
        return self.convolver.ff(self.bottom[0].get_act(), self.weight, self.bias)
    def bp(self):
        assert len(self.top) == 1 and len(self.bottom) == 1
        self.weightgrad = self.convolver.weight_grad(self.top[0].sen, self.bottom[0].get_act())
        self.biasgrad = self.convolver.bias_grad(self.top[0].sen)
        self.bottom[0].sen = self.convolver.bp(self.top[0].sen, self.weight)

    def update(nsamples, mom, lr, wd):
        self.weightdelta = mom * self.weightdelta - lr * self.blobs_lr[0] / nsamples * self.weightgrad - lr * self.blobs_lr[0] * self.blobs_wd[0] * self.weight
        self.weight += self.weightdelta
        self.weightgrad = None # reset the grad to None to free the space
        self.biasdelta = mom * self.biasdelta - lr * self.blobs_lr[1] / nsamples * self.biasgrad - lr * self.blobs_lr[1] * self.blobs_wd[1] * self.bias
        self.bias += self.biasdelta
        self.biasgrad = None # reset the grad to None to free the space

class PoolingConnection(Connection):
    def __init__(self, name, kernel_size, stride, pad, op):
        super(PoolingConnection, self).__init__(name)
        self.pooler = co.Pooler(kernel_size, kernel_size, stride, stride, pad, pad, op)
    def ff(self):
        assert len(self.bottom) == 1
        return self.pooler.ff(self.bottom[0].get_act())
    def bp(self):
        assert len(self.bottom) == 1 and len(self.top) == 1
        self.bottom[0].sen = self.pooler.bp(self.top[0].sen, self.top[0].pre_nonlinear, self.bottom[0].get_act())

class LRNConnection(Connection):
    def __init__(self, name, local_size, alpha, beta):
        super(LRNConnection, self).__init__(name)
        self.lrner = co.Lrner(local_size, alpha, beta)
    def ff(self):
        #TODO:didn't implemented
        scale = owl.zeros(self.bottom[0].get_act().shape)
        return self.lrner.ff(self.bottom[0].get_act(), scale)
    def bp(self):
        return self.top[0].sen()

class SoftMaxConnection(Connection):
    def __init__(self, name):
        super(SoftMaxConnection, self).__init__(name)
    def ff(self):
        assert len(self.bottom) == 2
        return co.softmax(self.bottom[0].get_act(), co.soft_op.instance)
    def bp(self):
        assert len(self.bottom) == 2
        self.bottom[0].sen = co.softmax(self.bottom[0].get_act(), co.soft_op.instance) - self.bottom[1].get_act()

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
        pass

class Net:
    def __init__(self):
        pass

    def all_layers(self):
        pass

    def all_connections(self):
        pass

    def all_input_layers(self):
        pass

    def all_output_layers(self):
        pass

    def toporder(self):
        pass

    def reverse_toporder(self):
        pass

    def init_random(self):
        for l in self.all_layers():
            l.init_random()

    def ff(self):
        for l in self.toporder():
            l.ff()

    def bp(self):
        for l in self.reverse_toporder():
            l.bp()
