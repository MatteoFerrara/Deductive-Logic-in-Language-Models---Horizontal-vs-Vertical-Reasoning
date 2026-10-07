
import os
import torch
import random
import numpy as np

from contextlib import nullcontext

def setup_environment(use_gpu = True, init_seed = 1337):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_gpu = use_gpu and torch.cuda.is_available()
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    if use_gpu: 
        print("Using Cuda device",torch.cuda.current_device()) 
        print(torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device = torch.device("cuda" if use_gpu else "cpu")
    device_type = 'cuda' if use_gpu else 'cpu' # for later use in torch.autocast

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # amp is automatic mixed precision package for PyTorch: see https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    compile = False # use PyTorch 2.0 to compile the model to be faster (not in Windows yet)
    
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    np.random.seed(init_seed)

    return {'dtype': dtype, 'device_type': device_type, 'device': device, 'ctx': ctx, 'compile': compile, 'init_seed': init_seed}