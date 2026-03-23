import torch
import time

# ============================================================
# Automatically picks GPU (mps) on Mac, CPU on other machines
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_label = "Mac M5 GPU (MPS)"
else:
    device = torch.device("cpu")
    device_label = "CPU"

print(f"Running on: {device_label}\n")

# Create two massive matrices (10000x10000)
print("Creating large matrices...")
a = torch.rand(10000, 10000).to(device)
b = torch.rand(10000, 10000).to(device)

# Warm up
torch.matmul(a, b)

# Run heavy matrix multiplication 20 times
print("Running 20x matrix multiplication (10000x10000)...")
start = time.time()

for i in range(20):
    result = torch.matmul(a, b)
    print(f"  Round {i+1}/20 done")


end = time.time()
elapsed = end - start

print(f"\n Done!")
print(f"Device      : {device_label}")
print(f"Total time  : {elapsed:.2f} seconds")
print(f"Per round   : {elapsed/20:.3f} seconds")