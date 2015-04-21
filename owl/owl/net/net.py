''' A package for implementing Caffe-like network structure using Owl APIs.

The package implements Caffe-like network structure with some minor differences. It uses Caffe-defined 
protobuf as core data structure so Caffe users could easily adapt to this. The package serves the purpose
of:

1. Quick deployment of neural network training using configure file.
2. Demonstrate the power of ``owl`` package (it takes only several hundreds LOC to implement Caffe and run it on dataflow engine).
'''

import numpy as np
import math
import Queue

import owl
import owl.elewise as ele
import owl.conv as co
from caffe import *

from netio import LMDBDataProvider
from netio import ImageListDataProvider
from netio import ImageWindowDataProvider

class ComputeUnit(object):
    ''' Interface for each compute unit.

    In ``owl.net``, the network is graph (in fact a DAG) that is composed of ``ComputeUnit`` s.
    ``ComputeUnit`` is a wrap-up of Caffe's ``layer`` abstraction, but is more
    general and flexible in its function sigature.

    :ivar caffe.LayerParameter params: layer parameter in Caffe's proto structure
    :ivar str name: name of the unit; the name must be unique
    :ivar btm_names: names of the bottom units
    :vartype btm_names: list str
    :ivar top_names: names of the top units
    :vartype top_names: list str
    :ivar list int out_shape:

    .. note::
        ``params``, ``name``, ``btm_names`` and ``top_names`` will be parsed from Caffe's network
        description file. ``out_shape`` should be set in :py:meth:`compute_size`

    '''
    def __init__(self, params):
        self.params = params
        self.name = params.name
        self.btm_names = []
        self.top_names = []
        self.out_shape = None
    def __str__(self):
        return 'N/A unit'
    def compute_size(self, from_btm, to_top):
        ''' Calculate the output size of this unit

        This function will be called before training during the ``compute_size`` phase.
        The ``compute_size`` phase is a feed-forward-like phase, during which each ``ComputeUnit``, rather than
        calculating the output tensor but calculating the output size (list int) for the top units. The
        size is usually used to calculate the weight and bias size for initialization.

        :param dict from_btm: input size from bottom units
        :param dict to_top: output size to top units

        .. seealso::
            :py:meth:`FullyConnection.compute_size`
            :py:meth:`ConvConnection.compute_size`
            :py:meth:`Net.compute_size`
        '''
        pass
    def forward(self, from_btm, to_top, phase):
        ''' Function for forward propagation

        This function will be called during forward-propagation. The function
        should take input in ``from_btm``, perform customized computation, and then
        put the result in ``to_top``. Both ``from_btm`` and ``to_top`` are ``dict`` type 
        where key is a ``str`` of name of the bottom/top units and value is an ``owl.NArray``
        served as input or output of the function.

        :param dict from_btm: input from bottom units
        :param dict to_top: output for top units
        :param str phase: name of the phase of the running. Currently either ``"TRAIN"`` or ``"TEST"``
        '''
        pass
    def backward(self, from_top, to_btm, phase):
        ''' Function for backward propagation

        This function will be called during backward-propagation. Similar to :py:meth:`forward`,
        The function should take input in ``from_top``, perform customized computation, and then
        put the result in ``to_btm``. The function also need to calculate the gradient (if any) and
        save them to the ``weightgrad`` field (see :py:meth:WeightedComputeUnit.weight_update).

        :param dict from_top: input from top units
        :param dict to_btm: output for top units
        :param str phase: name of the phase of the running. Currently either ``"TRAIN"`` or ``"TEST"``
        '''
        pass
    def weight_update(self, base_lr, base_weight_decay, momentum, batch_size):
        ''' Function for weight update

        This function will be called during weight update. 

        :param float base_lr: base learning rate
        :param float base_weight_decay: base weight decay
        :param float momentum: momentum value
        :param int batch_size: the size of the current minibatch
        '''
        pass

class ComputeUnitSimple(ComputeUnit):
    ''' An auxiliary class for :py:class:`ComputeUnit` that will only have one input unit and one output unit.
    '''
    def __init__(self, params):
        super(ComputeUnitSimple, self).__init__(params)
    def compute_size(self, from_btm, to_top):
        ''' Set the ``out_shape`` as the same shape of the input. Inherited classes could override this function.
        '''
        to_top[self.top_names[0]] = from_btm[self.btm_names[0]][:]
        self.out_shape = to_top[self.top_names[0]][:]
    def forward(self, from_btm, to_top, phase):
        ''' Transform the interface from multiple input/output to only one input/output function :py:meth:`ff`.
        '''
        to_top[self.top_names[0]] = self.ff(from_btm[self.btm_names[0]], phase)
        self.out = to_top[self.top_names[0]]
    def ff(self, act, phase):
        ''' Function for forward-propagation

        :param owl.NArray act: the activation from the bottom unit
        :param str phase: name of the phase of the running. Currently either ``"TRAIN"`` or ``"TEST"``
        :return: the activation of this unit
        :rtype: owl.NArray
        '''
        pass
    def backward(self, from_top, to_btm, phase):
        ''' Transform the interface from multiple input/output to only one input/output function :py:meth:`bp`.
        '''
        to_btm[self.btm_names[0]] = self.bp(from_top[self.top_names[0]])
    def bp(self, sen):
        ''' Function for backward-propagation

        :param owl.NArray sen: the sensitivity (or error derivative to the input) from the top unit
        :return: the sensitivity of this unit
        :rtype: owl.NArray
        '''
        pass

class WeightedComputeUnit(ComputeUnitSimple):
    ''' An auxiliary class for :py:class:`ComputeUnit` with weights

    :ivar owl.NArray weight: weight tensor
    :ivar owl.NArray weightdelta: momentum of weight
    :ivar owl.NArray weightgrad: gradient of weight
    :ivar owl.NArray bias: bias tensor
    :ivar owl.NArray biasdelta: momentum of bias
    :ivar owl.NArray biasgrad: gradient of bias
    :ivar blobs_lr: learning rate specific for this unit; a list of float represents: [weight_lr, bias_lr]
    :vartype blobs_lr: list float
    :ivar weight_decay: weight decay specific for this unit; a list of float represents: [weight_wd, bias_wd]
    :vartype weight_decay: list float
    '''
    def __init__(self, params):
        super(WeightedComputeUnit, self).__init__(params)
        # weights and bias
        self.weight = None
        self.weightdelta = None
        self.weightgrad = None
        self.bias = None
        self.biasdelta = None
        self.biasgrad = None
      
        self.in_shape = None

        # blob learning rate and weight decay
        self.blobs_lr = params.blobs_lr
        self.weight_decay = params.weight_decay
        if len(self.blobs_lr) == 0:
            self.blobs_lr = [1,1]
        if len(self.weight_decay) == 0:
            self.weight_decay = [1, 0]
    
    def compute_size(self, from_btm, to_top):
        pass 
   
    def init_weights_with_filler(self):
        ''' Init weights & bias. The function will be called during weight initialization.

        Currently, four types of initializers are supported: ``"constant", "gaussian", "uniform", "xavier"``.
        '''
        #init weight
        npweights = None
        if self.weight_filler.type == "constant":
            npweights = np.ones(self.wshape, dtype = np.float32) * self.weight_filler.value
        elif self.weight_filler.type == "gaussian":
            npweights = np.random.normal(self.weight_filler.mean, self.weight_filler.std, self.wshape)
        elif self.weight_filler.type == "uniform":
            npweights = np.random.uniform(self.weight_filler.min, self.weight_filler.max, self.wshape)
        elif self.weight_filler.type == "xavier":
            fan_in = np.prod(self.in_shape[:])
            scale = np.sqrt(float(3)/fan_in)
            npweights = np.random.uniform(-scale, scale, self.wshape)
        self.weight = owl.from_numpy(npweights.astype(np.float32)).reshape(self.wshape)
      
        #init bias
        npwbias = None
        if self.bias_filler.type == "constant":
            npbias = np.ones(self.bshape, dtype = np.float32) * self.bias_filler.value
        elif self.bias_filler.type == "gaussian":
            npbias = np.random.normal(self.bias_filler.mean, self.bias_filler.std, self.bshape)
        elif self.bias_filler.type == "uniform":
            npbias = np.random.uniform(self.bias_filler.min, self.bias_filler.max, self.bshape)
        elif self.bias_filler.type == "xavier":
            fan_in = np.prod(self.in_shape[:])
            scale = np.sqrt(float(3)/fan_in)
            npbias = np.random.uniform(-scale, scale, self.bshape)
        self.bias = owl.from_numpy(npbias.astype(np.float32)).reshape(self.bshape)
        
    def weight_update(self, base_lr, base_weight_decay, momentum, batch_size):
        ''' Update the weight & bias

        Using following formula:

        ``$_delta = momentum * $_delta - (base_lr * $_lr / batch_size) * $_grad - (base_lr * $_lr * base_wd * $_wd) * $``
        
        , where ``$`` could be either ``weight`` or ``bias``.
        '''
        if self.weightdelta == None:
            self.weightdelta = owl.zeros(self.weightgrad.shape)

        self.weightdelta = momentum * self.weightdelta \
                        - (base_lr * self.blobs_lr[0] / batch_size) * self.weightgrad \
                        - (base_lr * self.blobs_lr[0] * base_weight_decay * self.weight_decay[0]) * self.weight
        self.weight = self.weight + self.weightdelta
        self.weightgrad = None

        if self.biasdelta == None:
            self.biasdelta = owl.zeros(self.biasgrad.shape)

        self.biasdelta = momentum * self.biasdelta \
                        - (base_lr * self.blobs_lr[1] / batch_size) * self.biasgrad \
                        - (base_lr * self.blobs_lr[1] * base_weight_decay * self.weight_decay[1]) * self.bias
        self.bias = self.bias + self.biasdelta
        self.biasgrad = None

class LinearUnit(ComputeUnitSimple):
    ''' Compute unit for linear transformation
    '''
    def ff(self, x, phase):
        return x
    def bp(self, y):
        return y
    def __str__(self):
        return 'linear'

class SigmoidUnit(ComputeUnitSimple):
    ''' Compute unit for Sigmoid non-linearity
    '''
    def ff(self, x, phase):
        return ele.sigm(x)
    def bp(self, y):
        return ele.sigm_back(y)
    def __str__(self):
        return 'sigmoid'

class ReluUnit(ComputeUnitSimple):
    ''' Compute unit for RELU non-linearity
    '''
    def ff(self, x, phase):
        self.ff_x = x
        return ele.relu(x)
    def bp(self, y):
        return ele.relu_back(y, self.ff_x)
    def __str__(self):
        return 'relu'

class TanhUnit(ComputeUnitSimple):
    ''' Compute unit for Hyperbolic Tangine non-linearity
    '''
    def ff(self, x, phase):
        return ele.tanh(x)
    def bp(self, y):
        return ele.tanh_back(y)
    def __str__(self):
        return 'tanh'

class PoolingUnit(ComputeUnitSimple):
    ''' Compute unit for Pooling

    .. note::
        The input and output is of size ``[HWCN]``:

        - ``H``: image height
        - ``W``: image width
        - ``C``: number of image channels (feature maps)
        - ``N``: size of minibatch
    '''
    def __init__(self, params):
        super(PoolingUnit, self).__init__(params)
        self.ppa = params.pooling_param
        if self.ppa.pool == PoolingParameter.PoolMethod.Value('MAX'):
            pool_ty = co.pool_op.max
        elif self.ppa.pool == PoolingParameter.PoolMethod.Value('AVE'):
            pool_ty = co.pool_op.avg
        self.pooler = co.Pooler(self.ppa.kernel_size, self.ppa.kernel_size,
                                self.ppa.stride, self.ppa.stride,
                                self.ppa.pad, self.ppa.pad,
                                pool_ty)
        
    def compute_size(self, from_btm, to_top):
        self.out_shape = from_btm[self.btm_names[0]][:]
        ori_height = self.out_shape[0]
        ori_width = self.out_shape[1]
        self.out_shape[0] = int(np.ceil(float(self.out_shape[0] + 2 * self.ppa.pad - self.ppa.kernel_size) / self.ppa.stride)) + 1
        self.out_shape[1] = int(np.ceil(float(self.out_shape[1] + 2 * self.ppa.pad - self.ppa.kernel_size) / self.ppa.stride)) + 1
        if self.ppa.pad:
            if (self.out_shape[0] - 1) * self.ppa.stride >= ori_height + self.ppa.pad:
                self.out_shape[0] = self.out_shape[0] - 1
                self.out_shape[1] = self.out_shape[1] - 1
        to_top[self.top_names[0]] = self.out_shape[:]

    def ff(self, x, phase):
        self.ff_x = x
        self.ff_y = self.pooler.ff(x)
        return self.ff_y
    def bp(self, y):
        return self.pooler.bp(y, self.ff_y, self.ff_x)
    def __str__(self):
        return 'pooling'

class DropoutUnit(ComputeUnitSimple):
    ''' Compute unit for dropout
    '''
    def __init__(self, params):
        super(DropoutUnit, self).__init__(params)
        self.scale = 1.0 / (1.0 - self.params.dropout_param.dropout_ratio)
        self.keep_ratio = 1 - self.params.dropout_param.dropout_ratio
    def ff(self, x, phase):
        ''' Foward function of dropout
        
        The dropout mask will not be multiplied if under ``"TEST"`` mode.
        '''
        self.dropmask = owl.randb(x.shape, self.keep_ratio)
        if phase == "TRAIN":
            return ele.mult(x, self.dropmask)*self.scale
        else:
            return x
        #for gradient test
        #return x
    def bp(self, y):
        return ele.mult(y, self.dropmask)*self.scale
        #for gradient test
        #return y
    def __str__(self):
        return 'dropout'

class SoftmaxUnit(ComputeUnit):
    ''' Compute unit for softmax
    '''
    def __init__(self, params):
        super(SoftmaxUnit, self).__init__(params)
        self.loss_weight = params.loss_weight
    
    def compute_size(self, from_btm, to_top):
        to_top[self.top_names[0]] = from_btm[self.btm_names[0]][:]
        self.out_shape = to_top[self.top_names[0]][:]
    
    def forward(self, from_btm, to_top, phase):
        to_top[self.top_names[0]] = co.softmax(from_btm[self.btm_names[0]], co.soft_op.instance)
        self.ff_y = to_top[self.top_names[0]]
        #turn label into matrix form
        nplabel = np.zeros([self.ff_y.shape[1], self.ff_y.shape[0]], dtype=np.float32)
        self.strlabel = from_btm[self.btm_names[1]]
        
        for i in range(len(self.strlabel)):
            nplabel[i, self.strlabel[i]] = 1
        self.y = owl.from_numpy(nplabel)
        
    def backward(self, from_top, to_btm, phase):
        if len(self.loss_weight) == 1:
            to_btm[self.btm_names[0]] = (self.ff_y - self.y)*self.loss_weight[0]
        else:
            to_btm[self.btm_names[0]] = (self.ff_y - self.y)

    def getloss(self):
        ''' Get the loss of the softmax (cross entropy)
        '''
        lossmat = ele.mult(ele.ln(self.ff_y), self.y)
        res = lossmat.sum(0).sum(1).to_numpy()
        return -res[0][0] / lossmat.shape[1]

    def __str__(self):
        return 'softmax'

class AccuracyUnit(ComputeUnit):
    ''' Compute unit for calculating accuracy

    .. note::
        In terms of Minerva's lazy evaluation, the unit is a **non-lazy** one since it gets the actual
        contents (accuracy) out of an ``owl.NArray``.
    '''
    def __init__(self, params):
        super(AccuracyUnit, self).__init__(params)
        self.acc = 0
        self.batch_size = 0

    def compute_size(self, from_btm, to_top):
        to_top[self.top_names[0]] = from_btm[self.btm_names[0]][:]
        self.out_shape = to_top[self.top_names[0]][:]
    
    def forward(self, from_btm, to_top, phase):
        predict = from_btm[self.btm_names[0]].argmax(0)
        ground_truth = owl.from_numpy(from_btm[self.btm_names[1]]).reshape(predict.shape)
        self.batch_size = from_btm[self.btm_names[0]].shape[1]
        correct = (predict - ground_truth).count_zero()
        self.acc = correct * 1.0 / self.batch_size

    def backward(self, from_top, to_btm, phase):
        pass

    def __str__(self):
        return 'accuracy'

class LRNUnit(ComputeUnitSimple):
    ''' Compute unit for LRN
    '''
    def __init__(self, params):
        super(LRNUnit, self).__init__(params)
        self.lrner = co.Lrner(params.lrn_param.local_size, params.lrn_param.alpha, params.lrn_param.beta)
        self.scale = None
    def ff(self, x, phase):
        self.ff_x = x
        self.scale = owl.zeros(x.shape)
        self.ff_y = self.lrner.ff(x, self.scale)
        return self.ff_y
    def bp(self, y):
        return self.lrner.bp(self.ff_x, self.ff_y, self.scale, y)
    def __str__(self):
        return 'lrn'

class ConcatUnit(ComputeUnit):
    ''' Compute unit for concatenation

    Concatenate input arrays along the dimension specified by Caffe's ``concat_dim_caffe``
    '''
    def __init__(self, params):
        super(ConcatUnit, self).__init__(params)
        self.concat_dim_caffe = params.concat_param.concat_dim
        self.slice_count = []

    def compute_size(self, from_btm, to_top):
        to_top[self.top_names[0]] = from_btm[self.btm_names[0]][:]
        self.concat_dim = len(from_btm[self.btm_names[0]]) - 1 - self.concat_dim_caffe
        for i in range(1, len(self.btm_names)):
            to_top[self.top_names[0]][self.concat_dim] = to_top[self.top_names[0]][self.concat_dim] + from_btm[self.btm_names[i]][self.concat_dim]
        self.out_shape = to_top[self.top_names[0]][:]

    def forward(self, from_btm, to_top, phase):
        narrays = []
        self.concat_dim = len(from_btm[self.btm_names[0]].shape) - 1 - self.concat_dim_caffe
        for i in range(len(self.btm_names)):
            narrays.append(from_btm[self.btm_names[i]])
            self.slice_count.append(from_btm[self.btm_names[i]].shape[self.concat_dim])
        to_top[self.top_names[0]] = owl.concat(narrays, self.concat_dim)
    def backward(self, from_top, to_btm, phase):
        st_off = 0
        for i in range(len(self.btm_names)):
            to_btm[self.btm_names[i]] = owl.slice(from_top[self.top_names[0]],
                                                  self.concat_dim,
                                                  st_off,
                                                  self.slice_count[i])
            st_off += self.slice_count[i]
    def __str__(self):
        return 'concat'

class FullyConnection(WeightedComputeUnit):
    ''' Compute unit for traditional fully connected layer
    '''
    def __init__(self, params):
        super(FullyConnection, self).__init__(params)
        self.inner_product_param = params.inner_product_param
        self.weight_filler = params.inner_product_param.weight_filler
        self.bias_filler = params.inner_product_param.bias_filler
    
    def compute_size(self, from_btm, to_top):
        ''' Compute the output size and also weight and bias size
        The weight size is ``[top_shape[0], btm_shape[0]]``; the bias size is ``[top_shape[0], 1]``
        (assume both ``top`` and ``btm`` are 2-dimensional array)
        '''
        shp = from_btm[self.btm_names[0]][:]
        if len(shp) > 2:
            self.in_shape = [np.prod(shp[0:-1], dtype=np.int32), shp[-1]]
        else:
            self.in_shape = shp
        to_top[self.top_names[0]] = self.in_shape[:]
        to_top[self.top_names[0]][0] = self.inner_product_param.num_output
        to_top[self.top_names[0]][1] = 1 
        self.out_shape = to_top[self.top_names[0]][:]
        self.wshape = [self.out_shape[0], self.in_shape[0]]
        self.bshape = [self.out_shape[0], 1]
    
    def ff(self, act, phase):
        shp = act.shape
        if len(shp) > 2:
            a = act.reshape([np.prod(shp[0:-1], dtype=np.int32), shp[-1]])
        else:
            a = act
        self.ff_act = act # save ff value
        if self.weight == None:
            self.init_weights_with_filler()
        return self.weight * a + self.bias

    def bp(self, sen):
        shp = self.ff_act.shape
        if len(shp) > 2:
            a = self.ff_act.reshape([np.prod(shp[0:-1], dtype=np.int32), shp[-1]])
        else:
            a = self.ff_act
        self.weightgrad = sen * a.trans()
        self.biasgrad = sen.sum(1)
        s = self.weight.trans() * sen
        if len(shp) > 2:
            s = s.reshape(shp)
        return s
    def __str__(self):
        return 'fc'

class ConvConnection(WeightedComputeUnit):
    ''' Convolution operation

    .. note::
        The input and output is of size ``[HWCN]``:

        - ``H``: image height
        - ``W``: image width
        - ``C``: number of image channels (feature maps)
        - ``N``: size of minibatch

    '''
    def __init__(self, params):
        super(ConvConnection, self).__init__(params)
        self.conv_params = params.convolution_param
        self.convolver = co.Convolver(self.conv_params.pad,
                self.conv_params.pad, self.conv_params.stride, self.conv_params.stride)
        self.num_output = params.convolution_param.num_output
        self.group = params.convolution_param.group
        #TODO: hack, we don't want to slice agian to use it into bp as a parameter
        self.group_data = []
        self.group_filter = []
        self.group_bias = []
        self.weight_filler = params.convolution_param.weight_filler
        self.bias_filler = params.convolution_param.bias_filler
    
    def compute_size(self, from_btm, to_top):
        ''' Compute the output size and also weight and bias size

        .. note::
            The weight(kernel) size is ``[HWCiCo]``; bias shape is ``[Co]``:

            - ``H``: kernel_height
            - ``W``: kernel_width
            - ``Ci``: number of input channels
            - ``Co``: number of output channels
        '''
        self.in_shape = from_btm[self.btm_names[0]][:]
        to_top[self.top_names[0]] = from_btm[self.btm_names[0]][:]
        to_top[self.top_names[0]][0] = (to_top[self.top_names[0]][0] + 2 * self.conv_params.pad - self.conv_params.kernel_size) / self.conv_params.stride + 1
        to_top[self.top_names[0]][1] = (to_top[self.top_names[0]][1] + 2 * self.conv_params.pad - self.conv_params.kernel_size) / self.conv_params.stride + 1
        to_top[self.top_names[0]][2] = self.num_output
        self.out_shape = to_top[self.top_names[0]][:]
        self.wshape = [self.conv_params.kernel_size,
                       self.conv_params.kernel_size,
                       self.in_shape[2],
                       self.num_output]
        self.bshape = [self.out_shape[2]]
    
    def ff(self, act, phase):
        ''' Feed-forward of convolution

        .. warning::
            Currently multi-group convolution (as in AlexNet paper) is not supported. One could walk around it by
            using a bigger convolution with number of feature maps doubled.
        '''
        if self.group == 1:
            self.ff_act = act
            if self.weight == None:
                self.init_weights_with_filler()
            return self.convolver.ff(act, self.weight, self.bias)
        else:
            #currently doesn't support multi-group
            assert(False)
        
    def bp(self, sen):
        ''' Backward propagation of convolution

        .. warning::
            Currently multi-group convolution (as in AlexNet paper) is not supported. One could walk around it by
            using a bigger convolution with number of feature maps doubled.
        '''
        if self.group == 1:
            self.weightgrad = self.convolver.weight_grad(sen, self.ff_act, self.weight)
            self.biasgrad = self.convolver.bias_grad(sen)
            return self.convolver.bp(sen, self.ff_act, self.weight)
        else:
            #currently doesn't support multi-group
            assert(False)
            
    def __str__(self):
        return 'conv'

class DataUnit(ComputeUnit):
    ''' The base class of dataunit.
    
    :ivar dp: dataprovider, different kind of dp load data from different formats
    :ivar generator: the iterator produced by dataprovider
    '''

    def __init__(self, params, num_gpu):
        super(DataUnit, self).__init__(params)

    def compute_size(self, from_btm, to_top):
        pass

    def forward(self, from_btm, to_top, phase):
        ''' Feed-forward of data unit will get a batch of a fixed batch_size from data provider. 

        .. note::
            
            Phase indicates whether it's training or testing. Usualy, the data augmentation operation for training involves some randomness, while testing doesn't
        
        '''
        
        if self.generator == None:
            self.generator = self.dp.get_mb(phase)

        while True:
            try:
                (samples, labels) = next(self.generator)
                if len(labels) == 0:
                    (samples, labels) = next(self.generator)
            except StopIteration:
                print 'Have scanned the whole dataset; start from the begginning agin'
                self.generator = self.dp.get_mb(phase)
                continue
            break

        to_top[self.top_names[0]] = owl.from_numpy(samples).reshape(
                [self.crop_size, self.crop_size, 3, samples.shape[0]])
        #may have multiplier labels
        for i in range (1, len(self.top_names)):
            to_top[self.top_names[i]] = labels[:,i - 1]
    def backward(self, from_top, to_btm, phase):
        # no bp pass
        pass
    def __str__(self):
        return 'data'

class LMDBDataUnit(DataUnit):
    ''' DataUnit load from LMDB.

    :ivar caffe.LayerParameter params: lmdb data layer param defined by Caffe, params.data_param contains information about data source, parmas.transform_param mainly defines data augmentation operations
    
    '''
    
    
    def __init__(self, params, num_gpu):
        super(LMDBDataUnit, self).__init__(params, num_gpu)
        if params.include[0].phase == Phase.Value('TRAIN'):
            self.dp = LMDBDataProvider(params.data_param, params.transform_param, num_gpu)
        else:
            self.dp = LMDBDataProvider(params.data_param, params.transform_param, 1)
        self.params = params
        self.crop_size = params.transform_param.crop_size
        self.generator = None

    def compute_size(self, from_btm, to_top):
        self.out_shape = [self.params.transform_param.crop_size,
                          self.params.transform_param.crop_size,
                          3, 1]
        to_top[self.top_names[0]] = self.out_shape[:]
   
    def forward(self, from_btm, to_top, phase):
        ''' Feed-forward operation may vary according to phase. 

        .. note::

            LMDB data provider now support multi-view testing, if phase is "MULTI_VIEW", it will produce concequtive 10 batches of different views of the same original image     
        '''
        if self.generator == None:
            if phase == 'TRAIN' or phase == 'TEST':
                self.generator = self.dp.get_mb(phase)
            #multiview test
            else:
                self.generator = self.dp.get_multiview_mb()
        while True:
            try:
                (samples, labels) = next(self.generator)
                if len(labels) == 0:
                    (samples, labels) = next(self.generator)
            except StopIteration:
                print 'Have scanned the whole dataset; start from the begginning agin'
                self.generator = self.dp.get_mb(phase)
                continue
            break
        to_top[self.top_names[0]] = owl.from_numpy(samples).reshape(
                [self.crop_size, self.crop_size, 3, samples.shape[0]])
        for i in range (1, len(self.top_names)):
            to_top[self.top_names[i]] = labels[:,i - 1]

    def __str__(self):
        return 'lmdb_data'

class ImageDataUnit(DataUnit):
    ''' DataUnit load from raw images.
    :ivar caffe.LayerParameter params: image data layer param defined by Caffe, this is often used when data is limited. Loading from original image will be slower than loading from LMDB
    '''
    
    def __init__(self, params, num_gpu):
        super(ImageDataUnit, self).__init__(params, num_gpu)
        if params.include[0].phase == Phase.Value('TRAIN'):
            self.dp = ImageListDataProvider(params.image_data_param, params.transform_param, num_gpu)
        else:
            self.dp = ImageListDataProvider(params.image_data_param, params.transform_param, 1)
        self.params = params
        self.crop_size = params.transform_param.crop_size
        self.generator = None

    def compute_size(self, from_btm, to_top):
        self.out_shape = [self.params.transform_param.crop_size,
                          self.params.transform_param.crop_size,
                          3, 1]
        to_top[self.top_names[0]] = self.out_shape[:]

    def __str__(self):
        return 'image_data'

class ImageWindowDataUnit(DataUnit):
    ''' DataUnit load from image window patches. 
    :ivar caffe.LayerParameter params: image window data layer param defined by Caffe, this is often used when data is limited and object bounding box is given

    '''
    
    def __init__(self, params, num_gpu):
        super(ImageWindowDataUnit, self).__init__(params, num_gpu)
        if params.include[0].phase == Phase.Value('TRAIN'):
            self.dp = ImageWindowDataProvider(params.window_data_param, num_gpu)
        else:
            self.dp = ImageWindowDataProvider(params.window_data_param, 1)
        self.params = params
        self.crop_size = params.window_data_param.crop_size
        self.generator = None
    
    #reset generator
    def reset_generator(self):
        if self.params.include[0].phase == Phase.Value('TRAIN'):
            self.generator = self.dp.get_mb('TRAIN')
        else:
            self.generator = self.dp.get_mb('TEST')

    def compute_size(self, from_btm, to_top):
        self.out_shape = [self.params.window_data_param.crop_size,
                          self.params.window_data_param.crop_size,
                          3, 1]
        to_top[self.top_names[0]] = self.out_shape[:]
    
    def __str__(self):
        return 'window_data'

class Net:
    ''' The class for neural network structure

    The Net is basically a graph (DAG), of which each node is a :py:class:`ComputeUnit`.

    :ivar units: all the ``ComputeUnit`` s.
    :vartype units: list owl.net.ComputeUnit
    :ivar adjacent: the adjacent list (units are represented by their name)
    :vartype adjacent: list list str
    :ivar reverse_adjacent: the reverse adjacent list (units are represented by their name)
    :vartype reverse_adjacent: list list str
    :ivar dict name_to_uid: a map from units' name to the unit object
    :ivar loss_uids: all the units for computing loss
    :vartype loss_uids: list int
    :ivar accuracy_uids: all the units for calculating accuracy
    :vartype accuracy_uids: list int
    '''
    def __init__(self):
        self.units = []
        self.adjacent = []
        self.reverse_adjacent = []
        self.base_lr = 0
        self.base_weight_decay = 0
        self.momentum = 0
        self.name_to_uid = {}
        self.loss_uids = []
        self.accuracy_uids = []

    def add_unit(self, unit):
        ''' Method for adding units into the graph

        :param owl.net.ComputeUnit unit: the unit to add
        '''
        uid = len(self.units)
        self.units.append(unit)
        self.adjacent.append([])
        self.reverse_adjacent.append([])
        if not unit.name in self.name_to_uid:
            self.name_to_uid[unit.name] = []
        self.name_to_uid[unit.name].append(uid)
        return uid

    def connect(self, u1, u2):
        ''' Method for connecting two units

        :param str u1: name of the bottom unit
        :param str u2: name of the top unit
        '''
        self.adjacent[u1].append(u2)
        self.reverse_adjacent[u2].append(u1)

    def get_units_by_name(self, name):
        ''' Get ``ComputeUnit`` object by its name

        :param str name: unit name
        :return: the compute unit object of that name
        :rtype: owl.net.ComputeUnit
        '''
        return [self.units[uid] for uid in self.name_to_uid[name]]

    def get_loss_units(self):
        ''' Get all ``ComputeUnit`` object for loss

        :return: all compute unit object for computing loss
        :rtype: list owl.net.ComputeUnit
        '''
        return [self.units[uid] for uid in self.loss_uids]

    def get_accuracy_units(self):
        ''' Get all ``ComputeUnit`` object for accuracy

        :return: all compute unit object for computing accuracy
        :rtype: list owl.net.ComputeUnit
        '''
        return [self.units[uid] for uid in self.accuracy_uids]

    def get_data_unit(self, phase = 'TRAIN'):
        ''' Get the ``ComputeUnit`` object for data loading

        :param str phase: phase name of the run
        :return: the compute unit object for loading data
        :rtype: owl.net.ComputeUnit
        '''
        data_units = self.name_to_uid['data']
        for du in data_units:
            if not self._is_excluded(du, phase):
                return self.units[du]

    def get_weighted_unit_ids(self):
        ''' Get ids for all :py:class:owl.net.WeightedComputeUnit

        :return: ids of all weighted compute unit
        :rtype: list int
        '''
        weights_id = []
        for i in xrange(len(self.units)):
            if isinstance(self.units[i], WeightedComputeUnit):
                weights_id.append(i)
        return weights_id

    def _is_excluded(self, unit, phase):
        p = self.units[unit].params
        return phase != None and len(p.include) != 0 and p.include[0].phase != Phase.Value(phase)

    def _toporder(self, phase = None):
        depcount = [len(inunits) for inunits in self.reverse_adjacent]
        queue = Queue.Queue()
        # remove dep from excluded units
        for unit in range(len(depcount)):
            if self._is_excluded(unit, phase):
                for l in self.adjacent[unit]:
                    depcount[l] -= 1
        # find start units
        for unit in range(len(depcount)):
            count = depcount[unit]
            if count == 0:
                queue.put(unit)
        # run
        while not queue.empty():
            unit = queue.get()
            if self._is_excluded(unit, phase):
                continue
            yield unit
            for l in self.adjacent[unit]:
                depcount[l] -= 1
                if depcount[l] == 0:
                    queue.put(l)

    def _reverse_toporder(self, phase = None):
        depcount = [len(outunits) for outunits in self.adjacent]
        queue = Queue.Queue()
        # remove dep from excluded units
        for unit in range(len(depcount)):
            if self._is_excluded(unit, phase):
                for l in self.reverse_adjacent[unit]:
                    depcount[l] -= 1
        # find start units
        for unit in range(len(depcount)):
            count = depcount[unit]
            if count == 0:
                queue.put(unit)
        # run
        while not queue.empty():
            unit = queue.get()
            if self._is_excluded(unit, phase):
                continue
            yield unit
            for l in self.reverse_adjacent[unit]:
                depcount[l] -= 1
                if depcount[l] == 0:
                    queue.put(l)

    def compute_size(self, phase = 'TRAIN'):
        ''' Perform the compute_size phase before running
        '''
        unit_to_tops = [{} for name in self.units]
        for u in self._toporder(phase):
            from_btm = {}
            for btm in self.reverse_adjacent[u]:
                from_btm.update(unit_to_tops[btm])
            self.units[u].compute_size(from_btm, unit_to_tops[u])
        #for u in self._toporder(phase):
            #print self.units[u].name
            #print self.units[u].out_shape
    
    def forward(self, phase = 'TRAIN'):
        ''' Perform the forward pass
        '''
        unit_to_tops = [{} for name in self.units]
        for u in self._toporder(phase):
            from_btm = {}
            for btm in self.reverse_adjacent[u]:
                from_btm.update(unit_to_tops[btm])
            self.units[u].forward(from_btm, unit_to_tops[u], phase)

    def backward(self, phase = 'TRAIN'):
        ''' Perform the backward pass
        '''
        unit_to_btms = [{} for name in self.units]
        for u in self._reverse_toporder(phase):
            from_top = {}
            for top in self.adjacent[u]:
                for keys in unit_to_btms[top]:
                    if keys in from_top:
                        from_top[keys] += unit_to_btms[top][keys]
                    else:
                        from_top[keys] = unit_to_btms[top][keys]
            self.units[u].backward(from_top, unit_to_btms[u], phase)

    def update(self, uid):
        ''' Update weights of one compute unit of the given uid

        :param int uid: id of the compute unit to update
        '''
        self.units[uid].weight_update(self.current_lr,
                                      self.base_weight_decay,
                                      self.momentum,
                                      self.batch_size)

    def weight_update(self):
        ''' Update weights for all units
        '''
        for i in range(len(self.units)):
            self.update(i)

    def __str__(self):
        ret = 'digraph G {\n'
        for uid in range(len(self.units)):
            ret += 'n' + str(uid) + ' [label="' + self.units[uid].name + '"]\n'
        for uid in range(len(self.units)):
            for nuid in self.adjacent[uid]:
                ret += 'n' + str(uid) + ' -> n' + str(nuid) + '\n'
        return ret + '}\n'
