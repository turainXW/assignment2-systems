import torch
import time
import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from typing import Dict, Optional
import json
import contextlib

@contextlib.contextmanager
def nvtx_range(name: str, enabled: bool = True):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

def benchmark_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    device: str = "cpu",
    enable_profiling: bool = False,
    mixed_precision: bool = False  # 新增参数
) -> Dict[str, float]:
    model.eval()
    
    # 准备 Autocast 上下文
    # 注意：BF16 仅在支持的 GPU（如 A100）上有效，否则会回退到 FP32
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            with nvtx_range("warmup_forward", enable_profiling):
                with autocast_ctx: # 注入混合精度
                    _ = model(input_ids)
            if device == "cuda":
                torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()

            with nvtx_range(f"forward_iter_{i}", enable_profiling):
                start = time.perf_counter()
                with autocast_ctx: # 注入混合精度
                    _ = model(input_ids)
                if device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

            times.append(end - start)

    return {
        "mean_time": sum(times) / len(times),
        "std_time": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "min_time": min(times),
        "max_time": max(times)
    }

def benchmark_forward_backward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    device: str = "cpu",
    enable_profiling: bool = False,
    mixed_precision: bool = False, # 新增参数
    record_memory: bool = False  # 新增一个参数用来控制是否开启内存分析
) -> Dict[str, float]:
    torch.cuda.reset_peak_memory_stats() # 开始前重置

    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()

    # Warmup
    for _ in range(warmup_iterations):
        with nvtx_range("warmup_train_step", enable_profiling):
            optimizer.zero_grad()
            with autocast_ctx: # 前向传播开启 autocast
                logits = model(input_ids)
                loss = logits.mean()
            loss.backward() # 反向传播会自动处理 autocast 状态
            optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
    if record_memory and device == "cuda":
        print("Starting memory history recording...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)


    # Benchmark
    times = []
    actual_iterations = 2 if record_memory else num_iterations

    for i in range(actual_iterations):
        if device == "cuda":
            torch.cuda.synchronize()

        with nvtx_range(f"train_iter_{i}", enable_profiling):
            start = time.perf_counter()

            optimizer.zero_grad()
            
            with autocast_ctx: # 注入混合精度进行 Forward
                with nvtx_range(f"forward_{i}", enable_profiling):
                    logits = model(input_ids)
                    loss = logits.mean()

                with nvtx_range(f"backward_{i}", enable_profiling):
                    loss.backward()

            with nvtx_range(f"optimizer_step_{i}", enable_profiling):
                optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

        
        times.append(end - start)

    if record_memory and device == "cuda":
        print("Dumping memory snapshot...")
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None) # 停止记录
        print("Memory snapshot saved as 'memory_snapshot.pickle'")
    
    peak_train_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Training Peak Memory: {peak_train_mem:.2f} MB")

    return {
        "mean_time": sum(times) / len(times),
        "std_time": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "min_time": min(times),
        "max_time": max(times),
        "peak_train_mem": peak_train_mem
    }

def get_model_memory_usage(model: torch.nn.Module, device: str = "cpu", mixed_precision: bool = False) -> Dict[str, float]:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

        autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()
        
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            dummy_input = torch.randint(0, model.vocab_size, (1, model.context_length), device=device)
            with autocast_ctx: # 测量混合精度下的峰值显存
                _ = model(dummy_input)
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return {
            "param_memory_mb": param_memory,
            "peak_memory_mb": peak_memory,
        }
    else:
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        return {"param_memory_mb": param_memory}

def run_benchmarks(
    vocab_size: int = 10000,
    context_length: int = 512,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    rope_theta: float = 10000.0,
    batch_size: int = 8,
    device: str = "cpu",
    num_iterations: int = 10,
    enable_profiling: bool = False,
    mixed_precision: bool = False, # 新增参数
    record_memory: bool = False # <--- 加上这一行
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Benchmarking BasicsTransformerLM | Mixed Precision: {'BF16' if mixed_precision else 'OFF'}")
    print(f"{'='*60}")

    with nvtx_range("model_initialization", enable_profiling):
        model = BasicsTransformerLM(
            vocab_size=vocab_size, context_length=context_length,
            d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    # 测量显存
    memory_stats = get_model_memory_usage(model, device, mixed_precision)
    print(f"\nMemory Usage: {memory_stats['peak_memory_mb']:.2f} MB (Peak)")

    # Forward
    forward_stats = benchmark_forward(
        model, input_ids, num_iterations, device=device, 
        enable_profiling=enable_profiling, mixed_precision=mixed_precision
    )
    print(f"Forward Pass: {forward_stats['mean_time']*1000:.2f} ms")

    # Forward + Backward
    fb_stats = benchmark_forward_backward(
        model, input_ids, num_iterations, device=device, 
        enable_profiling=enable_profiling, mixed_precision=mixed_precision, record_memory=record_memory
    )
    print(f"Forward+Backward: {fb_stats['mean_time']*1000:.2f} ms")

    return {"config": locals(), "results": {"forward": forward_stats, "forward_backward": fb_stats, "memory": memory_stats}}

def main():
    parser = argparse.ArgumentParser()
    # ... 保留原有的参数 ...
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--enable-profiling", action="store_true")
    # 新增混合精度开关
    parser.add_argument("--mixed-precision", action="store_true", help="Enable BF16 mixed precision")
    parser.add_argument("--record-memory", action="store_true", help="Record CUDA memory history and dump snapshot")



    args = parser.parse_args()

    run_benchmarks(
        vocab_size=args.vocab_size, context_length=args.context_length,
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, d_ff=args.d_ff,
        rope_theta=args.rope_theta, batch_size=args.batch_size,
        device=args.device, num_iterations=args.num_iterations,
        enable_profiling=args.enable_profiling,
        mixed_precision=args.mixed_precision, # 传递参数
        record_memory=args.record_memory 
    )

if __name__ == "__main__":
    main()
