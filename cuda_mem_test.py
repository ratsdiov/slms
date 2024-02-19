import torch
import gc


def print_cuda_memory_stats(msg):
    print("*** " + msg)
    print(
        "torch.cuda.memory_allocated: %fGB"
        % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.max_memory_allocated: %fGB"
        % (torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )


print_cuda_memory_stats("Before context loading")
device = "cuda"
dummy = torch.ones(1, device=device)  # Force GPU context loading
print_cuda_memory_stats("After context loading")

t = torch.ones(100_000_000, device=device)
print_cuda_memory_stats("After t allocation 1")

t = torch.ones(100_000_000, device=device)
print_cuda_memory_stats("After t allocation 2")

t = torch.ones(100_000_000, device=device)
print_cuda_memory_stats("After t allocation 3")

del t
print_cuda_memory_stats("After del t")

gc.collect()
print_cuda_memory_stats("After gc.collect")

torch.cuda.empty_cache()
print_cuda_memory_stats("After empty_cache")
