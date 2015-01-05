import owl
import owl.conv as co
import numpy as np
import demo_common

x = owl.randn([227, 227, 3, 256], 0.0, 1)
w = owl.randn([11, 11, 3, 96], 0.0, 0.1)
b = owl.zeros([96])
conv = co.Convolver(pad_h=0, pad_w=0, stride_v=4, stride_h=4)

y = conv.ff(x, w, b)
print y.to_numpy()
print y.shape

ex = conv.bp(y, w)
print ex.to_numpy()
print ex.shape
