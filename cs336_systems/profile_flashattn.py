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
        mask = torch.triu(torch.ones(L, L, device=Q.device), diagonal=1).bool()
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


    lens = [2**p for p in range(7, 17)]
    dims = [2**p for p in range(4, 7)]
    dtypes = [torch.bfloat16, torch.float32]

    results = []
    outdir = "flash_attn_profile"
    os.makedirs(outdir, exist_ok=True)
    pkl_path = os.path.join(outdir, "results.pkl")

    try:
        print(f"\n{'='*135}")
        print(f"BENCHMARKING FLASHATTENTION-2 vs REGULAR PYTORCH ON: {devname}")
        print(f"{'='*135}")
        header = (f"| {'Type':>4} | {'L':>5} | {'D':>3} | "
                  f"{'FA Fwd':>10} | {'PT Fwd':>10} | "
                  f"{'FA Bwd':>10} | {'PT Bwd':>10} | "
                  f"{'FA E2E':>10} | {'PT E2E':>10} | (ms)")
        print(header)
        print(f"|{'-'*6}|{'-'*7}|{'-'*5}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")

        for D, dtype in itertools.product(dims, dtypes):
            dt_str = "bf16" if dtype == torch.bfloat16 else "fp32"
            fa_oom = False
            pt_oom = False

            torch.cuda.memory._record_memory_history(max_entries=100000)


            for L in lens:
                B = 1
                Q = torch.randn((B, L, D), device="cuda", dtype=dtype, requires_grad=True)
                K = torch.randn((B, L, D), device="cuda", dtype=dtype, requires_grad=True)
                V = torch.randn((B, L, D), device="cuda", dtype=dtype, requires_grad=True)
                dO = torch.randn((B, L, D), device="cuda", dtype=dtype)

                def run_bench(func, is_already_oom, Q, K, V, dO):
                    if is_already_oom:
                        return {"f": float('nan'), "b": float('nan'), "e": float('nan')}, True
                    
                    try:
                        # 1. Benchmark Forward (无梯度模式，最省显存)
                        def fwd_fn():
                            with torch.no_grad():
                                return func(Q, K, V, True)
                        f_lat = bench_fn(fwd_fn)

                        # 2. Benchmark Backward
                        # 核心改进：必须在函数内部执行 Forward，才能产生新的计算图供 Backward 消耗
                        def bwd_fn():
                            # 这里的前向传播是为了给后向传播提供计算图
                            # 我们只需要测量 backward 那一行的耗时吗？
                            # 实际上，业界标准通常是测量 E2E，或者如下构造：
                            q, k, v = Q.detach().requires_grad_(), K.detach().requires_grad_(), V.detach().requires_grad_()
                            tmp_out = func(q, k, v, True)
                            tmp_out.backward(dO) # 这里不需要 retain_graph，因为每次循环都会重新生成图
                        
                        # 注意：这样测得的是 Fwd+Bwd 的总和，我们需要减去 Fwd
                        fb_lat = bench_fn(bwd_fn)
                        b_lat = max(0, fb_lat - f_lat)

                        # 3. Benchmark E2E (其实就是 fb_lat)
                        e_lat = fb_lat

                        # 测量完毕后清理显存
                        torch.cuda.empty_cache()
                        
                        return {"f": f_lat, "b": b_lat, "e": e_lat}, False

                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        return {"f": float('nan'), "b": float('nan'), "e": float('nan')}, True
                
                
                fa_res, fa_oom = run_bench(FlashAttentionTT.apply, fa_oom, Q, K, V, dO)
                pt_res, pt_oom = run_bench(regular_pytorch_attention, pt_oom, Q, K, V, dO)



                res = {
                    "Type": dt_str, "L": L, "D": D,
                    "fa_f": fa_res["f"], "fa_b": fa_res["b"], "fa_e": fa_res["e"],
                    "pt_f": pt_res["f"], "pt_b": pt_res["b"], "pt_e": pt_res["e"]
                }
                results.append(res)

                def fmt(x): return f"{x:10.5f}" if not math.isnan(x) else f"{'OOM':>10}"
                print(f"| {dt_str:>4} | {L:5d} | {D:3d} | "
                      f"{fmt(res['fa_f'])} | {fmt(res['pt_f'])} | "
                      f"{fmt(res['fa_b'])} | {fmt(res['pt_b'])} | "
                      f"{fmt(res['fa_e'])} | {fmt(res['pt_e'])} |")
            

            try:
                snapshot_path = str(D)+"d_"+dt_str+"snapshot.pickle"
                torch.cuda.memory._dump_snapshot(outdir+"/"+snapshot_path)
                print(f"Memory snapshot saved to: {snapshot_path}")
            except Exception as e:
                print(f"Failed to dump memory snapshot: {e}")
            torch.cuda.memory._record_memory_history(None)


          

        df = pd.DataFrame(results)
        df_final = df.copy()
        num_cols = ["fa_f", "pt_f", "fa_b", "pt_b", "fa_e", "pt_e"]
        for col in num_cols:
            df_final[col] = df_final[col].apply(lambda x: f"{x:.5f}" if not pd.isna(x) else "OOM")

        print(f"\n{'='*135}")
        print("ALL TESTS COMPLETED. FINAL TABLE SUMMARY:")
        print(df_final.to_markdown(index=False, tablefmt="github", stralign="right", numalign="right"))

    finally:


        # ---- 失败也画图/输出表格 ----
        if len(results) > 0:
            df = pd.DataFrame(results)

            # 输出当前已记录的表格
            df_show = df.copy()
            num_cols = ["fa_f", "pt_f", "fa_b", "pt_b", "fa_e", "pt_e"]
            for col in num_cols:
                df_show[col] = df_show[col].apply(lambda x: f"{x:.5f}" if not pd.isna(x) else "OOM")
            print("\nPARTIAL TABLE SUMMARY:")
            print(df_show.to_markdown(index=False, tablefmt="github", stralign="right", numalign="right"))

            # 画图：FA vs PT（Fwd/Bwd/E2E）按 L 分别对比
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                outdir = "flash_attn_profile"
                os.makedirs(outdir, exist_ok=True)

                for dtype in df["Type"].unique():
                    for D in sorted(df["D"].unique()):
                        sub = df[(df["Type"] == dtype) & (df["D"] == D)].copy()
                        if sub.empty:
                            continue
                        sub = sub.sort_values("L")

                        # 过滤 NaN（OOM）
                        def safe(vals):
                            return np.array([v if not math.isnan(v) else np.nan for v in vals], dtype=float)

                        Ls = sub["L"].values
                        fa_f = safe(sub["fa_f"])
                        pt_f = safe(sub["pt_f"])
                        fa_b = safe(sub["fa_b"])
                        pt_b = safe(sub["pt_b"])
                        fa_e = safe(sub["fa_e"])
                        pt_e = safe(sub["pt_e"])

                        plt.figure(figsize=(10, 6))
                        plt.plot(Ls, fa_f, label="FA Fwd")
                        plt.plot(Ls, pt_f, label="PT Fwd")
                        plt.plot(Ls, fa_b, label="FA Bwd")
                        plt.plot(Ls, pt_b, label="PT Bwd")
                        plt.plot(Ls, fa_e, label="FA E2E")
                        plt.plot(Ls, pt_e, label="PT E2E")
                        plt.xlabel("Sequence Length (L)")
                        plt.ylabel("Latency (ms)")
                        plt.title(f"FlashAttention vs PyTorch | dtype={dtype}, D={D}")
                        plt.legend()
                        plt.grid(True, linestyle="--", alpha=0.4)

                        fig_path = os.path.join(outdir, f"plot_dtype-{dtype}_D-{D}.png")
                        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                        plt.close()
                        print(f"Saved plot: {fig_path}")
            except Exception as e:
                print(f"Plotting skipped due to error: {e}")

if __name__ == "__main__":
    main()
