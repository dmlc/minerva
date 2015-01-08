import owl
import numpy as np
import demo_common as dc

x1 = owl.randn([784, 128], 0.0, 0.1)
x2 = owl.randn([784, 128], 0.0, 0.1)
w = owl.randn([512, 784], 0.0, 0.1)
b = owl.zeros([512, 1])

y1 = w * x1 + b
y2 = w * x2 + b
gw = y1 * x1.trans() + y2 * x2.trans()
print gw.to_numpy()
