import owl
import owl.conv
import sys
owl.initialize(sys.argv)
cpu = owl.create_cpu_device()
gpu = [owl.create_gpu_device(i) for i in range(owl.get_gpu_device_count())]
print '''
     __   __   _   __   _   _____   ____    _    _   ___
    /  | /  | | | |  \\ | | |  ___| |  _ \\  | |  / / /   |
   /   |/   | | | |   \\| | | |__   | |_| | | | / / / /| |
  / /|   /| | | | |      | |  __|  |    /  | |/ / / /_| |
 / / |  / | | | | | |\\   | | |___  | |\\ \\  |   / / ___  |
/_/  |_/  |_| |_| |_| \\__| |_____| |_| \\_\\ |__/ /_/   |_|
'''
print '[INFO] You have %d GPU devices' % len(gpu)
print '[INFO] Set device to gpu[0]'
owl.set_device(gpu[0])
