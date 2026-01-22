import torch
import time
import argparse
import contextlib
import os
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from torch.autograd.profiler import emit_nvtx


def run_memory_profile(args, context_length):
    device = args.device
    mixed_precision = args.mixed_precision
    
    print(f"\n>>> Profiling Model | Context: {context_length} | MP: {mixed_precision} | Device: {device}")
    
    # 1. 动态初始化模型
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    
    # 打印参数量确认
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, context_length), device=device)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()

    # --- 阶段 0: 预热 (Warmup) ---
    # 预热一轮，让 CUDA 初始化算法，确保后续性能测试准确
    print("Warming up...")
    optimizer.zero_grad()
    with autocast_ctx:
        loss = model(input_ids).mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    # --- 阶段 1: Nsight Systems 性能分析 (nsys) ---
    print("Capturing NVTX ranges for nsys...")
    torch.cuda.cudart().cudaProfilerStart() 
    
    with emit_nvtx(record_shapes=True): 
        # 1.1 前向传播
        torch.cuda.nvtx.range_push("Forward_Pass")
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.mean()
        torch.cuda.synchronize() 
        torch.cuda.nvtx.range_pop()
        
        # 1.2 反向传播
        torch.cuda.nvtx.range_push("Backward_Pass")
        loss.backward()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        # 1.3 优化器更新
        torch.cuda.nvtx.range_push("Optimizer_Step")
        optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()
    print("NVTX capture finished.")


    # --- 阶段 2: 仅前向传播显存快照 (Memory Snapshot) ---
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    
    # 开启记录显存历史
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    with torch.no_grad():
        with autocast_ctx:
            _ = model(input_ids)
    
    fname_fwd = f"mem_fwd_ctx{context_length}_d{args.d_model}_l{args.num_layers}_mp{mixed_precision}.pickle"
    torch.cuda.memory._dump_snapshot(fname_fwd)
    peak_fwd = torch.cuda.max_memory_allocated() / (1024**2)
    
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Forward Peak: {peak_fwd:.2f} MB saved to {fname_fwd}")


    # --- 阶段 3: 完整训练步显存快照 (Forward + Backward + Optimizer) ---
    model.train()
    torch.cuda.reset_peak_memory_stats()
    
    # 重新开启记录显存历史
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    try:
        optimizer.zero_grad()
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.mean()
        loss.backward()
        optimizer.step()
        
        fname_full = f"mem_full_ctx{context_length}_d{args.d_model}_l{args.num_layers}_mp{mixed_precision}.pickle"
        torch.cuda.memory._dump_snapshot(fname_full)
        peak_full = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Full Step Peak: {peak_full:.2f} MB saved to {fname_full}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"!!! OOM for Context {context_length} during Full Training Step !!!")
            peak_full = -1
        else:
            raise e
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

    # 彻底清理内存，为下一个 context length 腾空间
    del model
    del optimizer
    torch.cuda.empty_cache()
    
    return peak_fwd, peak_full

def main():
    parser = argparse.ArgumentParser(description="Dynamic Memory Profiler for Transformer")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed-precision", action="store_true")

    args = parser.parse_args()

    results = []
    for cl in args.context_lengths:
        fwd, full = run_memory_profile(args, cl)
        results.append((cl, fwd, full))

    print("\n\n" + " SUMMARY TABLE ".center(50, "="))
    print(f"{'Context':<10} | {'Fwd Peak (MB)':<15} | {'Full Peak (MB)':<15}")
    print("-" * 50)
    for cl, fwd, full in results:
        full_str = f"{full:.2f}" if full > 0 else "OOM"
        print(f"{cl:<10} | {fwd:<15.2f} | {full_str:<15}")

if __name__ == "__main__":
    main()
