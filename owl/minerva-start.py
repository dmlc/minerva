import owl
import sys
owl.initialize(sys.argv)
cpu = owl.create_cpu_device()
gpu0 = owl.create_gpu_device(0)
gpu1 = owl.create_gpu_device(1)
print '''
     __   __   _   __   _   _____   ____    _    _   ___
    /  | /  | | | |  \\ | | |  ___| |  _ \\  | |  / / /   |
   /   |/   | | | |   \\| | | |__   | |_| | | | / / / /| |
  / /|   /| | | | |      | |  __|  |    /  | |/ / / /_| |
 / / |  / | | | | | |\\   | | |___  | |\\ \\  |   / / ___  |
/_/  |_/  |_| |_| |_| \\__| |_____| |_| \\_\\ |__/ /_/   |_|
'''
owl.set_device(gpu0)
print "[INFO] Set device to gpu0"
