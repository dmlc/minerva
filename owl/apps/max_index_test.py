from owl import *
import sys

def printArray(a):
    for i in range(a.shape()[0]):
        for j in range(a.shape()[1]):
            print a.tolist()[j * a.shape()[0] + i],
        print

if __name__ == '__main__':
    initialize(sys.argv)
    set_device(create_gpu_device(0))
    print "Example:"
    a = randn([3, 2], 0, 1)
    printArray(a)

    print "Max on 0:"
    b = a.max_index(0)
    printArray(b)

    print "Max on 1:"
    b = a.max_index(1)
    printArray(b)

    print "Another example:"
    a = make_narray([4, 3], [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
    printArray(a)

    print "Max on 0:"
    b = a.max_index(0)
    printArray(b)

    print "Max on 1:"
    b = a.max_index(1)
    printArray(b)

