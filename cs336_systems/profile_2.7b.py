import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
import argparse

# 确保能导入模型定义
try:
    from cs336_basics.model import BasicsTransformerLM, TransformerBlock
    from cs336_basics.optimizer import AdamW
except ImportError:
    print("导入失败！请确保当前目录或 PYTHONPATH 包含 cs336_basics 文件夹")
    raise

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_profiling(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # 混合精度设置
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if args.mixed_precision else torch.float32,
        reduce_dtype=torch.bfloat16 if args.mixed_precision else torch.float32,
        buffer_dtype=torch.bfloat16 if args.mixed_precision else torch.float32,
    )

    for cl in args.context_lengths:
        if rank == 0:
            print(f"\n>>> MODE: {args.mode} | CTX: {cl} | Mixed-Precision: {args.mixed_precision}")

        # 1. 实例化模型（在 CPU 上实例化以节省显存）
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=cl,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
        )

        # 2. FSDP 包装策略
        my_auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock}, 
        )

        # 3. 使用 FSDP 包装
        # sync_module_states=True 会在各 rank 间同步权重并移至 GPU
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True, 
        )

        optimizer = AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, cl), device=device)

        # --- 预热阶段 (Warmup) ---
        model.train()
        output = model(input_ids)
        if args.mode == "train":
            loss = output.mean()
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # --- 显存快照记录开始 ---
        if rank == 0:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
            torch.cuda.cudart().cudaProfilerStart()

        # --- 核心性能分析步 ---
        # 即使是 inference，我们也通过 model.train() 来保持显存分配的一致性（或根据需要改用 eval）
        if args.mode == "train":
            model.train()
            output = model(input_ids)
            loss = output.mean()
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                output = model(input_ids)

        torch.cuda.synchronize()

        # --- 结束采集并保存结果 ---
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStop()
            # 文件名：snapshot_{mode}_ctx{cl}_mp{True/False}.pickle
            filename = f"snapshot_{args.mode}_ctx{cl}_mp{args.mixed_precision}.pickle"
            torch.cuda.memory._dump_snapshot(filename)
            
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"[Rank 0] {args.mode.upper()} Peak Memory: {peak_mb:.2f} MB")
            print(f"[Rank 0] Saved snapshot: {filename}")
            torch.cuda.memory._record_memory_history(enabled=None)

        # 清理以便进行下一个 context length
        del model, optimizer
        torch.cuda.empty_cache()

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    # 2.7B 模型参数定义
    parser.add_argument("--mode", choices=["inference", "train"], required=True, help="运行模式")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--mixed-precision", action="store_true")
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPUs. Starting FSDP profiling...")
    mp.spawn(run_profiling, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
