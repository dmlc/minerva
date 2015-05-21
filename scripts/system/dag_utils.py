import sys

class Dag:
    def __init__(self, fname):
        self.fname = fname
        self.node_attr = {}
        self.adj = {}
        self.rev_adj = {}
    def load(self):
        with open(self.fname, 'r') as f:
            line = f.readline() # line == 'Nodes:'
            line = f.readline()
            while not line.startswith('Edges:'):
                [name, attr] = line.strip().split('>>>>')
                self.node_attr[name] = {pair.split('===')[0] : pair.split('===')[1] for pair in attr.split(';;;')[0:-1]}
                self.adj[name] = []
                self.rev_adj[name] = []
                line = f.readline()

            line = f.readline()
            while len(line.strip()) != 0:
                [src, dst] = line.strip().split(' -> ')
                self.adj[src].append(dst)
                self.rev_adj[dst].append(src)
                line = f.readline()

if __name__ == '__main__':
    dag = Dag(sys.argv[1])
    dag.load()
