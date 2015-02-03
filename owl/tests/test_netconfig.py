import sys
import owl.net as net
from owl.net_helper import CaffeNetBuilder

builder = CaffeNetBuilder(sys.argv[1], sys.argv[2])
owl_net = net.Net()
builder.build_net(owl_net)

for u in owl_net._toporder():
    print str(u) + '|' + owl_net.units[u].name, ' ',
print '>>>>>>>>>>>>>>>>'
print '>>>>>>>>>>>>>>>>'
for u in owl_net._toporder('TRAIN'):
    print str(u) + '|' + owl_net.units[u].name, ' ',
print '>>>>>>>>>>>>>>>>'
print '>>>>>>>>>>>>>>>>'
for u in owl_net._toporder('TEST'):
    print str(u) + '|' + owl_net.units[u].name, ' ',
print '>>>>>>>>>>>>>>>>'
print '>>>>>>>>>>>>>>>>'

for u in owl_net._reverse_toporder():
    print str(u) + '|' + owl_net.units[u].name, ' ',
print '<<<<<<<<<<<<<<<<'
print '<<<<<<<<<<<<<<<<'
for u in owl_net._reverse_toporder('TRAIN'):
    print str(u) + '|' + owl_net.units[u].name, ' ',
print '<<<<<<<<<<<<<<<<'
print '<<<<<<<<<<<<<<<<'
for u in owl_net._reverse_toporder('TEST'):
    print str(u) + '|' + owl_net.units[u].name, ' ',
print '<<<<<<<<<<<<<<<<'
print '<<<<<<<<<<<<<<<<'
