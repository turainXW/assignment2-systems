import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import json
from cs336_basics.model import BasicsTransformerLMByTriton

# --- 1. 配置参数 ---
OUTPUT_DIR = "ddp_profile"
D_MODELS = [768, 1024, 1536, 2048] # 不同的模型维度测试点
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 512,
    "num_layers": 12,
    "d_ff": 4096, # 动态调整时会基于 d_model * 4
    "rope_theta": 10000.0
}

# --- 2. Overlap DDP 实现 ---
class OverlapDDP(torch.nn.Module):
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
        for h in self.handles:
            h.wait()
        self.handles = []
        with torch.no_grad():
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad.data /= self.world_size

# --- 3. 辅助函数 ---
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29509"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# --- 4. 核心测试逻辑 ---
def benchmark_rank(rank, world_size, d_model, mode, results_dict):
    setup(rank, world_size)
    
    # 动态构建配置
    config = BASE_CONFIG.copy()
    config["d_model"] = d_model
    config["d_ff"] = d_model * 4
    config["num_heads"] = d_model // 64 

    
    # 初始化模型
    try:
        raw_model = BasicsTransformerLMByTriton(**config).to(torch.bfloat16).cuda(rank)
    except torch.cuda.OutOfMemoryError:
        if rank == 0: print(f"OOM at d_model={d_model}")
        cleanup(); return

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    model = OverlapDDP(raw_model) if mode == "overlap" else raw_model
    if mode != "overlap":
        with torch.no_grad():
            for p in model.parameters(): dist.broadcast(p.data, src=0)

    # 数据准备
    inputs = torch.randint(0, config["vocab_size"], (2, config["context_length"])).cuda(rank)
    targets = torch.randint(0, config["vocab_size"], (2, config["context_length"])).cuda(rank)

    start_step = torch.cuda.Event(enable_timing=True)
    end_step = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = criterion(model(inputs).view(-1, config["vocab_size"]), targets.view(-1))
        loss.backward()
        if mode == "overlap": model.finish_sync()
        else:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None: dist.all_reduce(p.grad.data)
        optimizer.step()

    # 测量 (5次取平均)
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start_step.record()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = criterion(model(inputs).view(-1, config["vocab_size"]), targets.view(-1))
        loss.backward()
        
        if mode == "overlap":
            model.finish_sync()
        elif mode == "batched":
            with torch.no_grad():
                grads = [p.grad.data for p in model.parameters() if p.grad is not None]
                flat_grad = torch.cat([g.view(-1) for g in grads])
                dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
                flat_grad /= world_size
                offset = 0
                for g in grads:
                    numel = g.numel(); g.copy_(flat_grad[offset:offset+numel].view_as(g)); offset += numel
        else: # naive
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                        p.grad.data /= world_size
        
        optimizer.step()
        end_step.record()
        torch.cuda.synchronize()
        times.append(start_step.elapsed_time(end_step))

    if rank == 0:
        results_dict[f"{mode}_{d_model}"] = float(np.mean(times))
    cleanup()

# --- 5. 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    manager = mp.Manager()
    raw_results = manager.dict()
    world_size = 2
    modes = ["naive", "batched", "overlap"]

    for d_model in D_MODELS:
        for mode in modes:
            print(f"Testing d_model={d_model}, mode={mode}...")
            mp.spawn(benchmark_rank, args=(world_size, d_model, mode, raw_results), nprocs=world_size, join=True)

    # 处理结果用于绘图
    plot_data = {mode: [] for mode in modes}
    for mode in modes:
        for d_model in D_MODELS:
            val = raw_results.get(f"{mode}_{d_model}", None)
            if val: plot_data[mode].append(val)

    # 1. 保存数据
    with open(os.path.join(OUTPUT_DIR, "scaling_data.json"), "w") as f:
        json.dump(dict(raw_results), f, indent=4)

    # 2. 绘图
    plt.figure(figsize=(10, 6))
    colors = {'naive': 'red', 'batched': 'blue', 'overlap': 'green'}
    markers = {'naive': 'o', 'batched': 's', 'overlap': '^'}

    for mode in modes:
        actual_d_models = D_MODELS[:len(plot_data[mode])] # 防止OOM导致数据缺失
        plt.plot(actual_d_models, plot_data[mode], label=mode.upper(), 
                 color=colors[mode], marker=markers[mode], linewidth=2, markersize=8)

    plt.xlabel('d_model size')
    plt.ylabel('Total Iteration Time (ms)')
    plt.title('DDP Performance Scaling across Model Sizes')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # 标注数据点
    for mode in modes:
        for i, val in enumerate(plot_data[mode]):
            plt.text(D_MODELS[i], val + 2, f"{val:.1f}", ha='center', fontsize=9)

    plt.savefig(os.path.join(OUTPUT_DIR, "scaling_comparison.png"))
    print(f"\nBenchmark finished. Check {OUTPUT_DIR} for scaling_comparison.png")
