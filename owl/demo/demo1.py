import owl
import owl.elewise as ele
import numpy as np
import demo_common

x = owl.randn([784, 256], 0.0, 0.01)
w = owl.randn([512, 784], 0.0, 0.01)
b = owl.zeros([512, 1])

y = ele.relu(w * x + b)
print y.tolist()[0:10]

e = owl.randn([512, 256], 0.0, 0.01)
ey = ele.relu_back(e, y)
ex = ey * x.trans()
print ex.tolist()[0:10]
