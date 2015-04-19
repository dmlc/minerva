cimport minerva
import sys
from libc.stdlib cimport calloc, free
from libc.string cimport strcpy

def initialize():
    cdef int argc = len(sys.argv)
    cdef char** argv = <char**>(calloc(argc, sizeof(char*)))
    for i in range(argc):
        argv[i] = <char*>(calloc(len(sys.argv[i]) + 1, sizeof(char)))
        strcpy(argv[i], sys.argv[i])
    minerva.Initialize(&argc, &argv)
    for i in range(argc):
        free(argv[i])
    free(argv)

def finalize():
    minerva.Finalize()

def create_cpu_device():
    return minerva.CreateCpuDevice()

def create_gpu_device(i):
    return minerva.CreateGpuDevice(i)

def get_gpu_device_count():
    return minerva.GetGpuDeviceCount()

def wait_for_all():
    minerva.WaitForAll()

def set_device(i):
    minerva.SetDevice(i)

