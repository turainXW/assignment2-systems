import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 设置环境
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501" # 换一个端口防止冲突
    dist.init_process_group(backend, rank=rank, world_size=world_size)

# 2. 核心测试逻辑
def work(rank, world_size, num_size_mb, backend, results_dict):
    try:
        setup(rank, world_size, backend)
        
        # 确定设备
        if backend == "nccl":
            device = torch.device(f"cuda:0") # 只有一块GPU，强制指向0
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        # 创建数据 (float32 占 4 bytes)
        num_elements = (num_size_mb * 1024 * 1024) // 4
        data = torch.randn(num_elements, device=device)

        # --- 最佳实践 1: 预热 ---
        for _ in range(5):
            dist.all_reduce(data, async_op=False)

        # --- 最佳实践 2: 同步并计时 ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        iters = 10
        for _ in range(iters):
            dist.all_reduce(data, async_op=False)
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) / iters

        if rank == 0:
            results_dict[(backend, world_size, num_size_mb)] = elapsed_time
            print(f"Done: {backend}, WS={world_size}, Size={num_size_mb}MB -> {elapsed_time:.4f}s")

    except Exception as e:
        if rank == 0:
            print(f"Failed: {backend}, WS={world_size}, Size={num_size_mb}MB. Error: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def plot_results(df):
    # 创建文件夹
    output_dir = "parallel"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # 创建两个子图：Gloo (CPU) 和 NCCL (GPU)
    backends = df['Backend'].unique()
    fig, axes = plt.subplots(1, len(backends), figsize=(15, 6), sharey=True)
    
    if len(backends) == 1:
        axes = [axes]

    for i, backend in enumerate(backends):
        sub_df = df[df['Backend'] == backend]
        ax = axes[i]
        sns.lineplot(data=sub_df, x="Size (MB)", y="Avg Time (s)", hue="World Size", 
                     marker="o", ax=ax, palette="viridis")
        ax.set_title(f"Backend: {backend.upper()}")
        ax.set_xscale('log') # 因为数据量跨度大，用对数坐标
        ax.set_yscale('log')
        ax.set_ylabel("Average Time (seconds)")
        ax.set_xlabel("Data Size (MB)")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "all_reduce_benchmark.png")
    plt.savefig(plot_path)
    print(f"\n[Success] Plot saved to: {plot_path}")

def run_benchmarks():
    configs = {
        "backends": ["gloo", "nccl"], 
        "world_sizes": [2, 4, 6],
        "sizes_mb": [1, 10, 100, 1000]
    }
    
        manager = mp.Manager()
        results_dict = manager.dict()

    for backend in configs["backends"]:
        if backend == "nccl" and not torch.cuda.is_available():
            continue

        for ws in configs["world_sizes"]:
            for size in configs["sizes_mb"]:
                mp.spawn(
                    work,
                    args=(ws, size, backend, results_dict),
                    nprocs=ws,
                    join=True
                )

    # 转换结果并绘图
    df_data = []
    for (b, w, s), t in results_dict.items():
        df_data.append({"Backend": b, "World Size": w, "Size (MB)": s, "Avg Time (s)": t})
    
    if df_data:
        df = pd.DataFrame(df_data)
        print("\n" + "="*30)
        print(df.to_string(index=False))
        plot_results(df)
    else:
        print("No data collected. Check errors above.")

if __name__ == "__main__":
    # 在 Windows 上运行 mp.spawn 需要这个
    run_benchmarks()
