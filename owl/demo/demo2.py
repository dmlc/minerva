import owl
import owl.conv as co
import numpy as np
import demo_common

x = owl.randn([227, 227, 3, 256], 0.0, 1)
w = owl.randn([11, 11, 3, 96], 0.0, 0.1)
b = owl.zeros([96])
conv = co.Convolver(0, 0, 4, 4) #pad_h, pad_w, stride_h, stride_w

y = conv.ff(x, w, b)
print y.shape
y.start_eval()
y.wait_for_eval()

ex = conv.bp(y, w)
print ex.shape
ex.start_eval()
ex.wait_for_eval()
