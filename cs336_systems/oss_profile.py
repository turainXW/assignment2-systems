import os
import time
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from typing import Type, Any, Optional

# 导入你的模型 (假设 BasicsTransformerLMByTriton 在你的路径中)
from cs336_basics.model import BasicsTransformerLMByTriton
# 导入你实现的优化器分片类 (如果不在一个文件，请修改 import)
from optimizer_state_sharding import optimizer_state_sharding

# --- 1. XL 模型配置 ---
XL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 512,
    "num_layers": 24,    # XL 规模
    "d_model": 2048,     # XL 规模
    "d_ff": 8192,
    "num_heads": 32,
    "rope_theta": 10000.0
}

# --- 2. 这里是你之前提供的 OverlapDDP 和 optimizer_state_sharding 类 ---
# 为了脚本能跑，请确保这里包含你之前写的那些类定义（由于篇幅原因此处略过，运行时请合并）

class OverlapDDP(torch.nn.Module):
    # ... (你提供的代码) ...
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook(p))
    
    def _make_hook(self, p):
        def hook(*args):
            handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)
        return hook

    def forward(self, *args, **kwargs):
        self.handles = []
        return self.module(*args, **kwargs)

    def finish_sync(self):
        for h in self.handles: h.wait()
        self.handles = []
        with torch.no_grad():
            for p in self.module.parameters():
                if p.grad is not None: p.grad.data /= self.world_size

# ----------------------------------------------------------------------------

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29510"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_mem_gb():
    return torch.cuda.max_memory_allocated() / (1024**3)

def benchmark_sharding(rank, world_size, use_sharding, results_dict):
    setup(rank, world_size)
    torch.cuda.reset_peak_memory_stats()
    
    # 1. 初始化模型 (bfloat16)
    raw_model = BasicsTransformerLMByTriton(**XL_CONFIG).to(torch.bfloat16).cuda()
    model = OverlapDDP(raw_model)
    
    mem_init = get_mem_gb() # 记录点 A: 模型初始化后

    # 2. 初始化优化器 (AdamW)
    if use_sharding:
        # 使用你实现的包装器
        optimizer = optimizer_state_sharding(model.parameters(), torch.optim.AdamW, lr=1e-4)
    else:
        # 标准优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 模拟数据
    inputs = torch.randint(0, XL_CONFIG["vocab_size"], (2, XL_CONFIG["context_length"])).cuda()
    targets = torch.randint(0, XL_CONFIG["vocab_size"], (2, XL_CONFIG["context_length"])).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    # 预热 & 测量
    times = []
    mem_before = 0
    mem_after = 0

    for i in range(5):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = criterion(model(inputs).view(-1, XL_CONFIG["vocab_size"]), targets.view(-1))
        loss.backward()
        model.finish_sync()
        
        if i == 1: mem_before = get_mem_gb() # 记录点 B: Step 之前 (此时有梯度)
        
        optimizer.step()
        
        if i == 1: mem_after = get_mem_gb()  # 记录点 C: Step 之后 (此时有状态)

        torch.cuda.synchronize()
        if i > 0: times.append(time.perf_counter() - start_time)

    if rank == 0:
        mode = "sharded" if use_sharding else "normal"
        results_dict[mode] = {
            "init": mem_init,
            "before_step": mem_before,
            "after_step": mem_after,
            "time": np.mean(times)
        }
    cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    OUTPUT_DIR = "opti_profile"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    manager = mp.Manager()
    final_results = manager.dict()
    world_size = 2

    # 分别测试两种模式
    for sharding_flag in [False, True]:
        print(f"Running benchmark with sharding={sharding_flag}...")
        mp.spawn(benchmark_sharding, args=(world_size, sharding_flag, final_results), nprocs=world_size, join=True)

    # --- 绘图逻辑 ---
    res_n = final_results["normal"]
    res_s = final_results["sharded"]

    # 1. 显存对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    stages = ['After Init', 'Before Step (Grads)', 'After Step (States)']
    x = np.arange(len(stages))
    ax1.bar(x - 0.2, [res_n['init'], res_n['before_step'], res_n['after_step']], 0.4, label='Normal Optimizer')
    ax1.bar(x + 0.2, [res_s['init'], res_s['before_step'], res_s['after_step']], 0.4, label='Sharded Optimizer')
    ax1.set_ylabel('Peak Memory (GB)')
    ax1.set_title('Peak Memory Usage: Normal vs Sharded (XL Model)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.legend()

    # 2. 速度对比图
    modes = ['Normal', 'Sharded']
    times = [res_n['time'], res_s['time']]
    ax2.bar(modes, times, color=['blue', 'orange'])
    ax2.set_ylabel('Time per Iteration (s)')
    ax2.set_title('Training Speed Overhead')
    for i, v in enumerate(times): ax2.text(i, v, f"{v:.3f}s", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sharding_performance.png"))
    
    # 打印回答问题所需的数据
    print("\n--- Final Accounting Results ---")
    print(f"Normal: Init={res_n['init']:.2f}G, BeforeStep={res_n['before_step']:.2f}G, AfterStep={res_n['after_step']:.2f}G, Time={res_n['time']:.3f}s")
    print(f"Sharding: Init={res_s['init']:.2f}G, BeforeStep={res_s['before_step']:.2f}G, AfterStep={res_s['after_step']:.2f}G, Time={res_s['time']:.3f}s")
