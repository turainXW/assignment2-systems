import torch
from cs336_basics.nn_utils import softmax
import time
import os

# 确保输出目录存在
output_dir = "attn_profile"
os.makedirs(output_dir, exist_ok=True)

def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    # 使用 einsum 计算注意力分数 (B, L, L)
    attn_logits = torch.einsum("... i d, ... j d -> ... i j", q, k)
    # 缩放
    attn_logits = attn_logits / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=q.device))
    # Softmax
    attn_weights = softmax(attn_logits, dim=-1)
    # 计算输出 (B, L, d)
    values = torch.einsum("... i j, ... j d -> ... i d", attn_weights, v)
    return values

def profile_attention(d_model, context_length, batch_size=8):
    device = 'cuda'
    
    # 1. 清理缓存并重置显存峰值统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # 2. 在 GPU 上创建张量并开启梯度
        q = torch.randn(batch_size, context_length, d_model, device=device, requires_grad=True)
        k = torch.randn(batch_size, context_length, d_model, device=device, requires_grad=True)
        v = torch.randn(batch_size, context_length, d_model, device=device, requires_grad=True)

        compiled_attention = torch.compile(scaled_dot_product_attention)

        # 3. 预热 (Warm up)
        for _ in range(10):
            _ = compiled_attention(q, k, v)
        torch.cuda.synchronize()

        # --- 开始记录显存历史 ---


        # 4. 前向计时 (100次)
        start_fw = time.perf_counter()
        for _ in range(100):
            output = compiled_attention(q, k, v)
        torch.cuda.synchronize()
        fw_time = (time.perf_counter() - start_fw) / 100 * 1000 # 换算为 ms

        # 获取前向峰值内存 (MB)
        fw_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
        torch.cuda.memory._record_memory_history(max_entries=100000)
        # 5. 反向计时 (100次)
        loss = output.sum()
        start_bw = time.perf_counter()
        for _ in range(100):
            output = compiled_attention(q, k, v)
            loss = output.sum()
            loss.backward() # 不再需要 retain_graph=True
        torch.cuda.synchronize()

        bw_time = (time.perf_counter() - start_bw) / 100 * 1000 # 换算为 ms

        # 获取反向后的累计峰值内存 (MB)
        total_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # 保存反向快照到 attn_profile 文件夹
        fname_bw = os.path.join(output_dir, f"attn_compiled_ctx{context_length}_d{d_model}.pickle")
        torch.cuda.memory._dump_snapshot(fname_bw)

        # 停止历史记录
        torch.cuda.memory._record_memory_history(None)

        return fw_time, bw_time, total_mem_mb

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # 如果 OOM，记录快照以便后续分析具体是在哪一步爆的
            try:
                fname_oom = os.path.join(output_dir, f"OOM_ctx{context_length}_d{d_model}.pickle")
                torch.cuda.memory._dump_snapshot(fname_oom)
            except:
                pass
            return "OOM", "OOM", "OOM"
        else:
            raise e

def main():
    d_models = [16, 32, 64, 128]
    context_lengths = [256, 1024, 4096, 8192, 16384]

    # 打印表头
    print(f"{'d_model':>8} | {'Ctx_Len':>8} | {'FW (ms)':>10} | {'BW (ms)':>10} | {'MaxMem(MB)':>10}")
    print("-" * 65)

    for d in d_models:
        for L in context_lengths:
            fw, bw, mem = profile_attention(d, L)
            if fw == "OOM":
                print(f"{d:8d} | {L:8d} | {'OOM':>10} | {'OOM':>10} | {'OOM':>10}")
            else:
                print(f"{d:8d} | {L:8d} | {fw:10.2f} | {bw:10.2f} | {mem:10.2f}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA not found. Please run on a machine with a GPU.")
    else:
        main()
