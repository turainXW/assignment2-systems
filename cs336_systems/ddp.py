import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from cs336_basics.model import BasicsTransformerLMByTriton

XL_CONFIG={
    "vocab_size": 50257,
    "context_length": 512,
    "d_model": 1536,   # 修改这里：1024 -> 768
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 6144,     # 建议：1536 * 4
    "rope_theta": 10000.0
}

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505" # 换一个端口防止冲突
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def benchmark_rank(rank,world_size):
    setup(rank, world_size)
    model=BasicsTransformerLMByTriton(**XL_CONFIG).to(torch.bfloat16).cuda(rank)

    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    criterion=torch.nn.CrossEntropyLoss()

    for param in model.parameters():
        dist.broadcast(param.data,src=0)

    batch_size=2

    inputs=torch.randint(0,XL_CONFIG["vocab_size"],(batch_size,XL_CONFIG["context_length"])).cuda(rank)
    targets=torch.randint(0,XL_CONFIG["vocab_size"],(batch_size,XL_CONFIG["context_length"])).cuda(rank)

    start_step=torch.cuda.Event(enable_timing=True)
    end_step=torch.cuda.Event(enable_timing=True)
    start_comm=torch.cuda.Event(enable_timing=True)
    end_comm=torch.cuda.Event(enable_timing=True)

    for warmup in range(5):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, XL_CONFIG["vocab_size"]), targets.view(-1))
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    grad_tensor = param.grad.data.contiguous()
                    dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
                    param.grad.data = grad_tensor / world_size
            # grads=[param.grad.data.contiguous() for param in model.parameters() if param.grad is not None]

            # from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
            # flat_grads=_flatten_dense_tensors(grads)
            # dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
            # flat_grads /= world_size
            # for old_grad, new_grad in zip(grads, _unflatten_dense_tensors(flat_grads, grads)):
            #     old_grad.copy_(new_grad)
        optimizer.step()


    torch.cuda.synchronize()
    start_step.record()

    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs=model(inputs)
        loss=criterion(outputs.view(-1,XL_CONFIG["vocab_size"]),targets.view(-1))
    loss.backward()


    start_comm.record()
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                grad_tensor = param.grad.data.contiguous()
                dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
                param.grad.data = grad_tensor / world_size
        # grads=[param.grad.data.contiguous() for param in model.parameters() if param.grad is not None]
        # from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        # flat_grads=_flatten_dense_tensors(grads)
        # dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        # flat_grads /= world_size
        # for old_grad, new_grad in zip(grads, _unflatten_dense_tensors(flat_grads, grads)):
        #     old_grad.copy_(new_grad)
    end_comm.record()
    optimizer.step()
    end_step.record()
    torch.cuda.synchronize()

    # 输出结果
    if rank == 0:
        total_ms = start_step.elapsed_time(end_step)
        comm_ms = start_comm.elapsed_time(end_comm)
        print(f"\n--- Naive DDP Benchmark Results ---")
        print(f"Total Step Time: {total_ms:.2f} ms")
        print(f"Gradient Comm Time: {comm_ms:.2f} ms")
        print(f"Comm Overhead: {(comm_ms/total_ms)*100:.2f}%")
        print(f"Max Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    cleanup()

if __name__ == "__main__":
    world_size = 2 # 你的两张卡
    mp.spawn(benchmark_rank, args=(world_size,), nprocs=world_size, join=True)


