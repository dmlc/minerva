import owl
import sys

owl.initialize(sys.argv)
owl.create_cpu_device()
gpu0 = owl.create_gpu_device(0)
gpu1 = owl.create_gpu_device(1)
owl.set_device(gpu0)
