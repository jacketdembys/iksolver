
#import GPUtil
import time
import numpy as np
import torch
import py3nvml.py3nvml as nvml


# Initialize NVML, this will start the NVML library
def getstuff():
    nvml.nvmlInit()
    # Get the count of GPUs available
    deviceCount = nvml.nvmlDeviceGetCount()
    i=0
        # Get the handle for each GPU
    handle = nvml.nvmlDeviceGetHandleByIndex(i)
    # Get GPU name
    name = nvml.nvmlDeviceGetName(handle)
    # Get memory information
    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_ut=memory_info.used / 1024 ** 2
    total_gpu=memory_info.total / 1024 ** 2
    gpu_info = {
    'GPU Name': nvml.nvmlDeviceGetName(handle),
    'util': gpu_ut,
    'total': total_gpu,
    'Temperature (C)': nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
    }
    # Print information
    print(f"Device {i}: {name}")
    print(f"Memory Usage: {gpu_ut} MB of {total_gpu} MB")
    print(f"Temperature: {nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)} C")
    # Shutdown NVML, this will clean up any resources
    nvml.nvmlShutdown()
    return gpu_info

def increase_gpu_load(size):
    # Perform some GPU-intensive operations, like large matrix multiplications
    # This is just a dummy operation and may not effectively increase GPU load
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU operations cannot be performed.")
    # Set the size of the matrices
    size = size
    print(size)
    # Create random tensors and move them to the GPU
    a = torch.rand(size, size, device='cuda', dtype=torch.float32)
    b = torch.rand(size, size, device='cuda', dtype=torch.float32)
    # Perform matrix multiplication on the GPU
    c = torch.matmul(a, b)

def main():
    target_load = 0.9  # 90% GPU utilization
    size=5000
    while True:
        gpu_info=getstuff()
        util=gpu_info['util']/gpu_info['total']
        if util < target_load:
            print("Increasing load...")
            increase_gpu_load(size=size)
            size+=1000
            print(size)
        else:
            print("Target GPU load reached.")
            increase_gpu_load(size=size)
            #return
        time.sleep(5)  # Wait for 5 seconds before checking again

if __name__ == '__main__':
    main()












