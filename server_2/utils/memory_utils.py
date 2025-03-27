import psutil
import torch
import os
import gc


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024
pass

def get_peak_memory_usage(func, *args, **kwargs):
    device_param = kwargs.pop('device', None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start_mem = get_memory_usage()
    result = func(*args, **kwargs)
    end_mem = get_memory_usage()
    peak_gpu_mem = 0
    if torch.cuda.is_available() and device_param == "cuda":
        peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        return result, peak_gpu_mem
    else:
        return result, max(0, end_mem - start_mem)
pass