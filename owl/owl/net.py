import owl
import owl.elewise as ele
import owl.conv as co

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
        return ele.relu_back(y, ff_y)

class TahnNeuron(Neuron):
    def ff(self, x):
        return ele.tahn(x)
    def bp(self, y, ff_y, ff_x):
        return ele.tahn_back(y)

class PoolingNeuron(Neuron):
    def __init__(self, h, w, stride_v, stride_h, op):
        super(PoolingNeuron, self).__init__()
        self.pooler = co.Pooler(h, w, stride_v, stride_h, op)
    def ff(self, x):
        return self.pooler.ff(x)
    def bp(self, y, ff_y, ff_x):
        return self.pooler.bp(y, ff_y, ff_x)


class Layer(object):
    def __init__(self, neuron, dropout):
        self.neuron = neuron
        self.dropout = dropout
        self.pre_nonlinear = None
        self.act = None
        self.sen = None
        self.dropmask = None
        # connectivity
        self.ff_conns = []
        self.bp_conns = []

    def ff(self):
        if len(self.ff_conns) == 0:
            return # input layer, act is set externally
        self.act = None
        for conn in self.ff_conns:
            if self.act == None:
                self.act = conn.ff()
            else:
                self.act += conn.ff()
        if type(self.neuron) == PoolingNeuron: # to save memory, only store this for pooling neuron
            self.pre_nonlinear = self.act
        self.act = self.neuron.ff(self.act)
        if self.dropout != 0.0:
            self.dropmask = owl.randb(self.act.shape, self.dropout)
            self.act = ele.mult(self.act, self.dropmask)

    def bp(self):
        if len(self.bp_conns) == 0:
            return # output layer, sen is set externally
        self.sen = None
        for conn in self.bp_conns:
            if self.sen == None:
                self.sen = conn.bp()
            else:
                self.sen += conn.bp() #XXX correct ?
        if self.dropout != 0.0:
            self.sen = ele.mult(self.sen, self.dropmask)
            self.dropmask = None
        self.sen = self.neuron.bp(self.sen, self.act, self.pre_nonlinear)
        self.pre_nonlinear = None

class Connection(object):
    def __init__(self, wshape, bshape, winiter):
        self.weight = winiter(wshape)
        self.weightdelta = owl.zeros(wshape)
        self.weightgrad = None
        self.bias = owl.zeros(bshape)
        self.biasdelta = owl.zeros(bshape)
        self.biasgrad = None
        # connectivity
        self.lower = None
        self.higher = None

    def ff(self):
        pass

    def bp(self):
        pass

    def update(nsamples, mom, lr, wd):
        self.weightdelta = mom * self.weightdelta - lr / nsamples * self.weightgrad - lr * wd * self.weight
        self.weight += self.weightdelta
        self.weightgrad = None # reset the grad to None to free the space
        self.biasdelta = mom * self.biasdelta - lr / nsamples * self.biasgrad;
        self.bias += self.biasdelta
        self.biasgrad = None # reset the grad to None to free the space

class FullyConnection(Connection):
    def __init__(self, wshape, bshape, winiter):
        super(FullyConnection, self).__init__(wshape, bshape, winiter)
    def ff(self):
        shp = self.lower.act.shape
        if len(shp) > 2:
            a = self.lower.act.reshape([np.prod(shp[0:-1]), shp[-1]])
        else:
            a = self.lower.act
        return self.weight * a + self.bias
    def bp(self):
        self.weightgrad = self.higher.sen * self.lower.act.Trans()
        self.biasgrad = self.higher.sen.sum(0)
        s = self.weight.Trans() * self.higher.sen
        shp = self.lower.act.shape
        if len(shp) > 2:
            s = s.reshape(shp)
        return s

class ConvConnection(Connection):
    def __init__(self, wshape, bshape, winiter, pad_h, pad_w, stride_v, stride_h):
        super(ConvConnection, self).__init__(wshape, bshape, winiter)
        self.convolver = co.Convolver(pad_h, pad_w, stride_v, stride_h)
    def ff(self):
        return self.convolver.ff(self.lower.act, self.weight, self.bias)
    def bp(self):
        self.weightgrad = self.convolver.weight_grad(self.higher.sen, self.lower.act)
        self.biasgrad = self.convolver.bias_grad(self.higher.sen)
        return self.convolver.bp(self.higher.sen, self.weight)

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
