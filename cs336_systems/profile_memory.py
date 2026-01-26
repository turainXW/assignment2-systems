import torch
import time
import argparse
import os
import gc
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# --- 全局性能优化配置 ---
torch.set_float32_matmul_precision('high')  # 开启 TF32 加速矩阵乘法
import torch._functorch.config
torch._functorch.config.donated_buffer = False  # 兼容某些编译路径下的 backward

# 确保快照目录存在
OUTPUT_DIR = "transformer_profile"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_benchmark(args, context_length, use_compile=False):
    device = args.device
    mode_str = "Compiled" if use_compile else "Vanilla"
    print(f"\n>>> Running {mode_str} | Context: {context_length} | Device: {device}")

    # 1. 初始化模型
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, context_length), device=device)
    
    # 2. 应用 torch.compile
    if use_compile:
        print("Compiling model... (this may take several minutes)")
        model = torch.compile(model)

    try:
        # 3. 充分预热 (Warmup)
        # 对于编译模式，至少需要 10-15 次迭代让 Triton Kernel 稳定
        warmup_steps = 15 if use_compile else 5
        print(f"Warming up ({warmup_steps} steps)...")
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            # 使用混合精度以节省显存
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = logits.mean()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        # 4. 测量前向传播耗时 (Forward Pass)
        num_iters = 30
        print(f"Measuring Forward Pass ({num_iters} iterations)...")
        start_fwd = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(num_iters):
                    _ = model(input_ids)
        torch.cuda.synchronize()
        fwd_time_ms = (time.perf_counter() - start_fwd) / num_iters * 1000

        # 5. 测量完整步耗时 (FW + BW + Opt)
        print(f"Measuring Full Step ({num_iters} iterations)...")
        start_full = time.perf_counter()
        for _ in range(num_iters):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = logits.mean()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        full_time_ms = (time.perf_counter() - start_full) / num_iters * 1000

        # 6. 显存记录与快照
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=100000)
        
        # 跑一次完整的步来捕捉显存峰值
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = logits.mean()
        loss.backward()
        optimizer.step()
        
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        
        # 存入快照
        snap_name = f"{mode_str}_ctx{context_length}_d{args.d_model}.pickle"
        torch.cuda.memory._dump_snapshot(os.path.join(OUTPUT_DIR, snap_name))
        torch.cuda.memory._record_memory_history(None)

        return fwd_time_ms, full_time_ms, peak_mem_mb

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"!!! OOM Detected in {mode_str} Mode !!!")
            return "OOM", "OOM", "OOM"
        else:
            raise e
    finally:
        # 严格清理显存，防止模式间干扰
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=50257)
    # 24GB 显存建议参数 (d_model=1024, layers=24 是一个比较稳妥的规模)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = []

    for cl in args.context_lengths:
        # 1. 运行 Vanilla
        fwd_v, full_v, mem_v = run_benchmark(args, cl, use_compile=False)
        # 2. 运行 Compiled
        fwd_c, full_c, mem_c = run_benchmark(args, cl, use_compile=True)
        
        results.append({
            "ctx": cl,
            "fwd_v": fwd_v, "fwd_c": fwd_c,
            "full_v": full_v, "full_c": full_c,
            "mem_v": mem_v, "mem_c": mem_c
        })

    # 打印最终对比表
    print("\n" + "="*85)
    print(f"{'Ctx Len':<8} | {'FW Vanilla':<12} | {'FW Compile':<12} | {'Full Vanilla':<12} | {'Full Compile':<12}")
    print("-" * 85)
    for r in results:
        def fmt(val): return f"{val:.2f}" if isinstance(val, float) else val
        print(f"{r['ctx']:<8} | {fmt(r['fwd_v']):<12} | {fmt(r['fwd_c']):<12} | {fmt(r['full_v']):<12} | {fmt(r['full_c']):<12}")
    print("="*85)
    print(f"Memory snapshots saved in ./{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
