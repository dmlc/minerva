import owl
import owl.elewise as ele
import owl.conv as co
import numpy as np
import Queue
import caffe

class Neuron(object):
    def __init__(self, params = None):
        self.params = params
        pass

    def ff(self, x):
        pass

    def bp(self, y):
        pass

class LinearNeuron(Neuron):
    def ff(self, x):
        return x
    def bp(self, y):
        return y

class SigmoidNeuron(Neuron):
    def ff(self, x):
        return ele.sigm(x)
    def bp(self, y):
        return ele.sigm_back(y)

class ReluNeuron(Neuron):
    def ff(self, x):
        self.ff_x = x
        return ele.relu(x)
    def bp(self, y):
        return ele.relu_back(y, self.ff_x)

class TahnNeuron(Neuron):
    def ff(self, x):
        return ele.tahn(x)
    def bp(self, y):
        return ele.tahn_back(y)

class PoolingNeuron(Neuron):
    def __init__(self, params):
        super(PoolingNeuron, self).__init__(params)
        if params.pool == PoolingParameter.PoolMethod.Value('MAX'):
            pool_ty = co.pool_op.max
        elif params.pool == PoolingParameter.PoolMethod.Value('AVE'):
            pool_ty = co.pool_op.avg
        self.pooler = co.Pooler(params.kernel_size, params.kernel_size, params.stride, params.stride, pool_ty)
    def ff(self, x):
        self.ff_x = x
        self.ff_y = self.pooler.ff(x)
        return self.ff_y
    def bp(self, y):
        return self.pooler.bp(y, self.ff_y, self.ff_x)

class DropoutNeuron(Neuron):
    def __init__(self, params):
        super(DropoutNeuron, self).__init__(params)
    def ff(self, x):
        self.dropmask = owl.randb(x, self.params.dropout_ratio)
        return ele.mult(x, self.dropmask)
    def bp(self, y):
        return ele.mult(y, self.dropmask)

class SoftmaxNeuron(Neuron):
    def __init__(self, params):
        super(SoftmaxNeuron, self).__init__(params)
    def ff(self, x):
        self.ff_y = co.softmax(x, co.soft_op.instance)
    def bp(self, y):
        return ff_y - y

class Layer(object):
    def __init__(self, name, neurons):
        self.name = name
        self.neurons = neurons
        self.act = None
        self.sen = None
        
        # layer dimension
        self.dim = []

        # connectivity
        self.ff_conns = []
        self.bp_conns = []

    def ff(self):
        # do merged sum
        merged_ff = None
        for ffconn in self.ff_conns:
            if merged_ff == None:
                merged_ff = ffconn.ff()
            else:
                merged_ff += ffconn.ff()
        if merged_ff != None:
            self.act = merged_ff
        # perform nonlinear functions
        for neu in self.neurons:
            self.act = neu.ff(self.act)

    def bp(self):
        # do merged sum
        merged_bp = None
        for bpconn in self.bp_conns:
            if merged_bp == None:
                merged_bp = bpconn.bp()
            else:
                merged_bp += bpconn.bp()
        if merged_bp != None:
            self.sen = merged_bp
        # derivative of nonlinear function
        for neu in reversed(self.neurons):
            self.sen = neu.bp(self.sen)

class Connection(object):
    def __init__(self, name, params):
        self.name = name
        self.params = params
        # weights and bias
        self.weight = None
        self.weightdelta = None
        self.weightgrad = None
        self.bias = None
        self.biasdelta = None
        self.biasgrad = None
        # connectivity
        self.bottom = None
        self.top = None

    def ff(self):
        pass

    def bp(self):
        pass

    '''
    def update(self, nsamples, glob_param):
        mom = glob_param.
        self.weightdelta = mom * self.weightdelta - lr / nsamples * self.weightgrad - lr * wd * self.weight
        self.weight += self.weightdelta
        self.weightgrad = None # reset the grad to None to free the space
        self.biasdelta = mom * self.biasdelta - lr / nsamples * self.biasgrad - lr * wd * self.bias
        self.bias += self.biasdelta
        self.biasgrad = None # reset the grad to None to free the space
    '''

class FullyConnection(Connection):
    def __init__(self, name, params):
        super(FullyConnection, self).__init__(name, params)
        
    def ff(self):
        shp = self.bottom.act.shape
        if len(shp) > 2:
            a = self.bottom.act.reshape([np.prod(shp[0:-1]), shp[-1]])
        else:
            a = self.bottom[0].act
        return self.weight * a + self.bias
    def bp(self):
        self.weightgrad = self.top.sen * self.bottom.act.Trans()
        self.biasgrad = self.top.sen.sum(0)
        s = self.weight.Trans() * self.top.sen
        shp = self.bottom.act.shape
        if len(shp) > 2:
            s = s.reshape(shp)
        return s

class ConvConnection(Connection):
    def __init__(self, name, params):
        super(ConvConnection, self).__init__(name, params)
        self.convolver = co.Convolver(params.pad, params.pad, params.stride, params.stride)
    def ff(self):
        return self.convolver.ff(self.bottom.act, self.weight, self.bias)
    def bp(self):
        self.weightgrad = self.convolver.weight_grad(self.top.sen, self.bottom.act)
        self.biasgrad = self.convolver.bias_grad(self.top.sen)
        return self.convolver.bp(self.top.sen, self.weight)

class LRNConnection(Connection):
    def __init__(self, name, local_size, alpha, beta):
        super(LRNConnection, self).__init__(name)
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
    def ff(self):
        #TODO:didn't implemented
        return self.bottom[0].get_act() 
    def bp(self):
        return self.top[0].sen()

class ConcatConnection(Connection):
    def __init__(self, name, concat_dim):
        super(ConcatConnection, self).__init__(name)
        self.concat_dim = concat_dim
    def ff(self):
        pass    
    def bp(self):
        pass

class Net:
    def __init__(self):
        self.layers = {}
        self.connections = []
        self.adjacent = {}
        self.reverse_adjacent = {}

    def add_layer(self, name, layer):
        self.layers[name] = layer
        self.adjacent[name] = []
        self.reverse_adjacent[name] = []

    def connect(self, lname1, lname2, conn):
        self.adjacent[lname1].append(lname2)
        self.reverse_adjacent[lname2].append(lname1)
        self.connections.append(conn)
        l1 = self.layers[lname1]
        l2 = self.layers[lname2]
        l1.bp_conns.append(l2)
        l2.ff_conns.append(l1)
        conn.bottom = lname1
        conn.top = lname2

    def _toporder(self):
        depcount = {layer : len(inlayers) for layer, inlayers in self.reverse_adjacent.iteritems()}
        queue = Queue.Queue()
        for layer, count in depcount.iteritems():
            if count == 0:
                queue.put(layer)
        while not queue.empty():
            layer = queue.get()
            yield self.layers[layer]
            for l in self.adjacent[layer]:
                depcount[l] -= 1
                if depcount[l] == 0:
                    queue.put(l)

    def _reverse_toporder(self):
        depcount = {layer : len(outlayers) for layer, outlayers in self.adjacent.iteritems()}
        queue = Queue.Queue()
        for layer, count in depcount.iteritems():
            if count == 0:
                queue.put(layer)
        while not queue.empty():
            layer = queue.get()
            yield self.layers[layer]
            for l in self.reverse_adjacent[layer]:
                depcount[l] -= 1
                if depcount[l] == 0:
                    queue.put(l)

    def ff(self):
        for l in self._toporder():
            l.ff()

    def bp(self):
        for l in self._reverse_toporder():
            l.bp()

class TestLayer(Layer):
    def __init__(self, name):
        self.name = name
        self.ff_conns = []
        self.bp_conns = []
    def ff(self):
        print 'ff:', self.name
    def bp(self):
        print 'bp:', self.name

if __name__ == '__main__':
    net = Net()
    net.add_layer('l1', TestLayer('l1'))
    net.add_layer('l2', TestLayer('l2'))
    net.add_layer('l3', TestLayer('l3'))
    net.add_layer('l4', TestLayer('l4'))
    net.add_layer('l5', TestLayer('l5'))
    net.add_layer('l6', TestLayer('l6'))
    net.add_layer('l7', TestLayer('l7'))
    net.add_layer('l8', TestLayer('l8'))
    net.connect('l1', 'l2', 1)
    net.connect('l1', 'l3', 2)
    net.connect('l2', 'l4', 3)
    net.connect('l3', 'l5', 4)
    net.connect('l4', 'l5', 5)
    net.connect('l5', 'l6', 6)
    net.connect('l5', 'l7', 7)
    net.connect('l7', 'l8', 8)
    net.ff()
    net.bp()
