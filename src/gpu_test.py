#!/usr/bin/env python3
"""
gpu_test.py - isola em que passo a GPU (ROCm) trava.

Uso:
    python src/gpu_test.py               # teste normal
    python src/gpu_test.py --no-miopen   # desliga o MIOpen (convolucao nativa do PyTorch)
"""
import faulthandler
import os
import sys

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
faulthandler.enable()  # mostra em qual linha do Python ocorreu o crash

import torch

if "--no-miopen" in sys.argv:
    torch.backends.cudnn.enabled = False  # no ROCm, isso desliga o MIOpen
    print("MIOpen DESLIGADO", flush=True)

print("torch", torch.__version__, "| hip", torch.version.hip, flush=True)
print("gpu:", torch.cuda.get_device_name(0), flush=True)

a = torch.randn(256, 256, device="cuda")
print("1) matmul ok", float((a @ a).sum()), flush=True)

x = torch.randn(1, 3, 64, 64, device="cuda")
conv = torch.nn.Conv2d(3, 16, 3).cuda()
y = conv(x)
torch.cuda.synchronize()
print("2) conv forward fp32 ok", tuple(y.shape), flush=True)

y.sum().backward()
torch.cuda.synchronize()
print("3) conv backward fp32 ok", flush=True)

with torch.autocast("cuda", dtype=torch.float16):
    z = conv(x)
torch.cuda.synchronize()
print("4) conv fp16 ok", z.dtype, flush=True)

print("GPU OK para treino")