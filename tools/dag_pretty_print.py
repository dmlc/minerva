import sys
import Queue
from dag_utils import Dag

fname = sys.argv[1]
dag = Dag(fname)
dag.load()

name_to_comp = {n : '' for n in dag.node_attr}
remains = set(dag.node_attr.keys())

def get_device_id(n):
    n_ty = dag.node_attr[n]['type']
    if n_ty == 'd':
        return dag.node_attr[n]['device_id']
    else:
        succ = dag.adj[n][0]
        return dag.node_attr[succ]['device_id']

# topological order
visited = {n : False for n in dag.node_attr}
depcount = {n : len(prev) for n, prev in dag.rev_adj.iteritems()}
queue = Queue.Queue()
for n, c in depcount.iteritems():
    if c == 0:
        queue.put(n)
dev_to_comp = {}
while not queue.empty():
    dev_to_names = {}
    renew_dev = set()
    while not queue.empty():
        n = queue.get()
        if visited[n]:
            continue
        visited[n] = True
        dev = get_device_id(n)
        if not dev in dev_to_names:
            dev_to_names[dev] = []
        dev_to_names[dev].append(n)
        #renew_dev = set(dev_to_names.keys())
        for prev in dag.rev_adj[n]:
            prev_dev = get_device_id(prev)
            if prev_dev != dev:
                renew_dev.add(prev_dev)
                renew_dev.add(dev)
        '''
        for succ in dag.adj[n]:
            succ_dev = get_device_id(succ)
            if succ_dev != dev:
                renew_dev.add(succ_dev)
                renew_dev.add(dev)
        '''
    for dev, names in dev_to_names.iteritems():
        if not dev in dev_to_comp or dev in renew_dev:
            dev_to_comp[dev] = names[0]
        for n in names:
            name_to_comp[n] = dev_to_comp[dev]
            for succ in dag.adj[n]:
                depcount[succ] -= 1
                if depcount[succ] == 0:
                    queue.put(succ)

# comp graph
comp_to_names = {}
comp_adj = {}
comp_rev_adj = {}
for n, c in name_to_comp.iteritems():
    if not c in comp_to_names:
        comp_to_names[c] = []
        comp_adj[c] = set()
        comp_rev_adj[c] = set()
    comp_to_names[c].append(n)
for n in dag.node_attr:
    for succ in dag.adj[n]:
        if name_to_comp[n] != name_to_comp[succ]:
            comp_adj[name_to_comp[n]].add(name_to_comp[succ])
            comp_rev_adj[name_to_comp[succ]].add(name_to_comp[n])

# do some merge to reduce #nodes
def merge_to(c1, c2): # there is an edge c2->c1. the func then merge c1 to c2
    comp_adj[c2].remove(c1)
    comp_adj[c2].update(comp_adj[c1])
    comp_to_names[c2] += comp_to_names[c1]
    for succ in comp_adj[c1]:
        comp_rev_adj[succ].remove(c1)
        comp_rev_adj[succ].add(c2)
    # clear
    comp_to_names[c1] = []
    comp_adj[c1] = set()
    comp_rev_adj[c1] = set()

'''
depcount = {n : len(prev) for n, prev in comp_rev_adj.iteritems()}
queue = Queue.Queue()
for n, c in depcount.iteritems():
    if c == 0:
        queue.put(n)
dev_to_comp = {}
while not queue.empty():
    c = queue.get()
    succ_lst = comp_adj[c]
    dev = get_device_id(c)
    cand_merge = None
    for prev in comp_rev_adj[c]:
        if get_device_id(prev) == dev:
            succ_dev = dev
            for succ in comp_adj[prev]:
                if get_device_id(succ) != dev:
                    succ_dev = get_device_id(succ)
                    break
            if succ_dev == dev:
                cand_merge = prev
                break
    if cand_merge != None:
        tmp_rev = set(comp_rev_adj[c])
        tmp_rev.remove(cand_merge)
        merge_to(c, cand_merge)
        for p in tmp_rev:
            comp_adj[p].remove(c)
        break
    #print c, '|', tmp_rev, ' v.s. ', prev, '|', comp_rev_adj[prev]
    for succ in succ_lst:
        depcount[succ] -= 1
        if depcount[succ] == 0:
            queue.put(succ)
'''

#print {c : len(ns) for c,ns in comp_to_names.iteritems()}
#print comp_adj

# draw
def get_device_color(d):
    colors = ['red', 'blue', 'green', 'orange']
    return colors[int(d)]

def one_node_string(n):
    s = n + ' '
    n_ty = dag.node_attr[n]['type']
    if n_ty == 'd':
        s += '[shape=box,style=filled,label=\"\",color=' + get_device_color(get_device_id(n)) + ']'
    else:
        s += '[style=filled,color=' + get_device_color(get_device_id(n)) + ',label=\"' + dag.node_attr[n]['name'][0:6] + '\"]'
    return s

def get_size(n):
    min_size = 1
    max_size = 5
    min_num = 1
    max_num = 200
    return min_size + (max_size - min_size) * float(n - min_num) / (max_num - min_num)

def comp_node_string(c, ns):
    s = c + ' [shape=circle,style=bold'
    op_names = set()
    for n in ns:
        n_ty = dag.node_attr[n]['type']
        if n_ty == 'o':
            op_names.add(dag.node_attr[n]['name'][0:6])
    s += ',label=\"#' + str(len(ns)) + '\\n' + ';'.join(list(op_names)) + '\"'
    size = get_size(len(ns))
    s += ',height=' + str(size) + ',width=' + str(size)
    s += ',color=' + get_device_color(get_device_id(c))
    s += ']'
    return s

num_comps = 0
with open(fname + '.dag', 'w') as f:
    f.write('digraph G {\n')
    for c, ns in comp_to_names.iteritems():
        if len(ns) == 0:
            continue
        elif len(ns) == 1:
            f.write(one_node_string(ns[0]) + '\n')
        else:
            f.write(comp_node_string(c, ns) + '\n')
        num_comps += 1
    for c, adj in comp_adj.iteritems():
        for succ in adj:
            f.write(c + ' -> ' + succ + '\n')
    f.write('}\n')

print '#comp:', num_comps
