import torch
import triton
import math
import itertools
import pandas as pd
import os
import pickle
from datetime import datetime

# ----------------- 1. 导入/定义算子 -----------------
try:
    from flashAttn import FlashAttentionTT
except ImportError:
    # 仅作占位，实际运行需确保 flashAttn.py 存在
    class FlashAttentionTT:
        @staticmethod
        def apply(q, k, v, causal):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

def regular_pytorch_attention(Q, K, V, is_causal=True):
    """ 题目要求：Regular PyTorch implementation (not using FlashAttention) """
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    # 显式计算 S = (Q @ K^T) * scale
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        L = Q.shape[-2]
        # 显式生成 Mask (这是 OOM 的主要原因)
        mask = torch.ones((L, L), device=Q.device, dtype=torch.bool).triu(1)
        S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)

# ----------------- 2. 测试工具 -----------------
def bench_fn(fn, warmup=25, rep=100):
    """ 使用 triton 官方工具测量延迟 """
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)

def main():
    assert torch.cuda.is_available(), "Requires NVIDIA GPU"
    devname = torch.cuda.get_device_name(0)
    
    # 题目要求的参数范围
    lens = [2**p for p in range(7, 17)]  # 128 to 65536
    dims = [2**p for p in range(4, 8)]   # 16 to 128
    dtypes = [torch.bfloat16, torch.float32]
    
    results = []

    # 打印实时表头
    print(f"\n{'='*135}")
    print(f"BENCHMARKING FLASHATTENTION-2 vs REGULAR PYTORCH ON: {devname}")
    print(f"{'='*135}")
    header = (f"| {'Type':>4} | {'L':>5} | {'D':>3} | "
              f"{'FA Fwd':>10} | {'PT Fwd':>10} | "
              f"{'FA Bwd':>10} | {'PT Bwd':>10} | "
              f"{'FA E2E':>10} | {'PT E2E':>10} | (ms)")
    print(header)
    print(f"|{'-'*6}|{'-'*7}|{'-'*5}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")

    # 循环顺序：D 和 Type 在外，L 在内，方便 OOM 熔断
    for D, dtype in itertools.product(dims, dtypes):
        dt_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        
        # 针对当前 (D, dtype) 配置的 OOM 状态位
        fa_oom = False
        pt_oom = False

        for L in lens:
            B = 1
            # 准备随机输入
            Q = torch.randn((B, L, D), device="cuda", dtype=dtype, requires_grad=True)
            K = torch.randn((B, L, D), device="cuda", dtype=dtype, requires_grad=True)
            V = torch.randn((B, L, D), device="cuda", dtype=dtype, requires_grad=True)
            dO = torch.randn((B, L, D), device="cuda", dtype=dtype)

            # --- 内部测量逻辑 ---
            def run_bench(func, is_already_oom):
                if is_already_oom:
                    return {"f": float('nan'), "b": float('nan'), "e": float('nan')}, True
                
                try:
                    # 1. Forward 延迟
                    def fwd_fn(): 
                        with torch.no_grad(): func(Q, K, V, True)
                    f_lat = bench_fn(fwd_fn)

                    # 2. Backward 延迟
                    q, k, v = Q.detach().requires_grad_(), K.detach().requires_grad_(), V.detach().requires_grad_()
                    out = func(q, k, v, True)
                    def bwd_fn(): 
                        out.backward(dO, retain_graph=True)
                    b_lat = bench_fn(bwd_fn)

                    # 3. End-to-End (Fwd + Bwd) 延迟
                    def e2e_fn():
                        q1, k1, v1 = Q.detach().requires_grad_(), K.detach().requires_grad_(), V.detach().requires_grad_()
                        o = func(q1, k1, v1, True)
                        o.backward(dO)
                    e_lat = bench_fn(e2e_fn)
                    
                    return {"f": f_lat, "b": b_lat, "e": e_lat}, False
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    return {"f": float('nan'), "b": float('nan'), "e": float('nan')}, True

            # 测试两种实现
            fa_res, fa_oom = run_bench(FlashAttentionTT.apply, fa_oom)
            pt_res, pt_oom = run_bench(regular_pytorch_attention, pt_oom)

            # 数据持久化
            res = {
                "Type": dt_str, "L": L, "D": D,
                "fa_f": fa_res["f"], "fa_b": fa_res["b"], "fa_e": fa_res["e"],
                "pt_f": pt_res["f"], "pt_b": pt_res["b"], "pt_e": pt_res["e"]
            }
            results.append(res)

            # --- 实时打印当前行 (关键点) ---
            def fmt(x): return f"{x:10.5f}" if not math.isnan(x) else f"{'OOM':>10}"
            print(f"| {dt_str:>4} | {L:5d} | {D:3d} | "
                  f"{fmt(res['fa_f'])} | {fmt(res['pt_f'])} | "
                  f"{fmt(res['fa_b'])} | {fmt(res['pt_b'])} | "
                  f"{fmt(res['fa_e'])} | {fmt(res['pt_e'])} |")

    # --- 最终保存 ---
    outdir = "flash_attn_profile"
    os.makedirs(outdir, exist_ok=True)
    pkl_path = os.path.join(outdir, "results.pkl")
    
    # 转换为最终表格展示
    df = pd.DataFrame(results)
    df_final = df.copy()
    num_cols = ["fa_f", "pt_f", "fa_b", "pt_b", "fa_e", "pt_e"]
    for col in num_cols:
        df_final[col] = df_final[col].apply(lambda x: f"{x:.5f}" if not pd.isna(x) else "OOM")
    
    print(f"\n{'='*135}")
    print("ALL TESTS COMPLETED. FINAL TABLE SUMMARY:")
    print(df_final.to_markdown(index=False, tablefmt="github", stralign="right", numalign="right"))
    
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nPickle file saved to: {pkl_path}")

if __name__ == "__main__":
    main()
