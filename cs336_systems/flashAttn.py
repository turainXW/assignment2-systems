import torch
from torch import nn
import triton
import triton.language as tl
from cs336_basics import model
from torch import Tensor
from jaxtyping import Float, Bool, Int

def flash_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
    scale):
    
    # 初始化梯度张量，类型与输入保持一致 (可能是 bf16)
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    Bq, Bk = 64, 64
    N, Length, D = Q.shape
    assert Length % Bq == 0 and Length % Bk == 0
    Tq = Length // Bq
    Tk = Length // Bk
    
    # Delta term for dQ calculation
    Dvec = torch.sum(dO * O, dim=-1) # (N, Length)

    for i in range(0, Tk):
        k_tile = K[:, i*Bk:(i+1)*Bk, :]
        v_tile = V[:, i*Bk:(i+1)*Bk, :]
        
        # 临时累积器，防止精度不够
        dK_chunk = torch.zeros((N, Bk, D), device=Q.device, dtype=torch.float32)
        dV_chunk = torch.zeros((N, Bk, D), device=Q.device, dtype=torch.float32)

        for j in range(0, Tq):
            q_tile = Q[:, j*Bq:(j+1)*Bq, :]
            o_tile = O[:, j*Bq:(j+1)*Bq, :]
            do_tile = dO[:, j*Bq:(j+1)*Bq, :]
            
            dq_tile = torch.zeros((N, Bq, D), device=Q.device, dtype=torch.float32)
            
            l_tile = L[:, j*Bq:(j+1)*Bq]
            D_tile = Dvec[:, j*Bq:(j+1)*Bq]

            # Recompute Attention Score S
            # cast to float for precision in backward math
            q_tile_f = q_tile.float()
            k_tile_f = k_tile.float()
            v_tile_f = v_tile.float()
            
            s_tile = torch.einsum('...bd,...kd->...bk', q_tile_f, k_tile_f) * scale
            
            if is_causal:
                q_pos = torch.arange(j*Bq, (j+1)*Bq, device=Q.device)
                k_pos = torch.arange(i*Bk, (i+1)*Bk, device=Q.device)
                mask = (k_pos[None, :] <= q_pos[:, None])
                s_tile = s_tile.masked_fill(~mask[None, :, :], float('-inf'))

            # P = exp(S - L)
            p_tile = torch.exp(s_tile - l_tile.unsqueeze(-1)) # (N, Bq, Bk)
            
            # dV = P^T * dO
            # dV accum: (N, Bk, D)
            do_tile_f = do_tile.float()
            p_tile_f = p_tile.float() # ensure P is float
            
            dv_tile = torch.einsum('...qk,...qd->...kd', p_tile_f, do_tile_f)
            
            # dP = dO * V^T
            dp_tile = torch.einsum('...qd,...kd->...qk', do_tile_f, v_tile_f)
            
            # dS = P * (dP - D)
            ds_tile = p_tile_f * (dp_tile - D_tile.unsqueeze(-1)) * scale
            
            # dQ = dS * K
            dq_tile = torch.einsum('...qk,...kd->...qd', ds_tile, k_tile_f)
            
            # dK = dS^T * Q
            dk_tile = torch.einsum('...qk,...qd->...kd', ds_tile, q_tile_f)
            
            # Accumulate (handle mixed precision by casting before adding)
            dK_chunk += dk_tile
            dV_chunk += dv_tile
            dQ[:, j*Bq:(j+1)*Bq, :] += dq_tile.to(dQ.dtype)

        # Store K/V gradients
        dK[:, i*Bk:(i+1)*Bk, :] += dK_chunk.to(dK.dtype)
        dV[:, i*Bk:(i+1)*Bk, :] += dV_chunk.to(dV.dtype)
        
    return dQ, dK, dV


class FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq, Bk = 16, 16
        N, L, D = Q.shape
        assert L % Bq == 0 and L % Bk == 0
        Tq = L // Bq
        Tk = L // Bk

        out = torch.empty_like(Q)
        L_out = torch.empty((N, L), device=Q.device, dtype=torch.float32)

        for i in range(0, Tq):
            q_tile = Q[:, i*Bq:(i+1)*Bq, :]
            out_tile = torch.zeros((N, Bq, D), device=Q.device, dtype=Q.dtype)
            m_prev = torch.full((N, Bq), float('-inf'), device=Q.device, dtype=torch.float32)
            l_tile = torch.zeros((N, Bq), device=Q.device, dtype=torch.float32)
            for j in range(0, Tk):
                k_tile = K[:, j*Bk:(j+1)*Bk, :]
                v_tile = V[:, j*Bk:(j+1)*Bk, :]

                s_tile = torch.einsum('...bd,...kd->...bk', q_tile, k_tile) / (D**0.5)

                if is_causal:
                    q_pos = torch.arange(i * Bq, (i + 1) * Bq, device=Q.device)
                    k_pos = torch.arange(j * Bk, (j + 1) * Bk, device=Q.device)
                    mask = (k_pos[None, :] <= q_pos[:, None])
                    s_tile = s_tile.masked_fill(~mask[None, None, :, :], float('-inf'))

                m_ij = torch.max(s_tile, dim=-1).values
                m_new = torch.maximum(m_prev, m_ij)

                p_tile = torch.exp(s_tile - m_new.unsqueeze(-1))
                l_tile_new = torch.exp(m_prev - m_new) * l_tile + p_tile.sum(dim=-1)
                
                # Careful with shapes and broadcasting here in reference impl
                out_tile = torch.einsum('...bk,...kd->...bd', p_tile, v_tile) + \
                           torch.exp(m_prev - m_new).unsqueeze(-1) * out_tile

                l_tile = l_tile_new
                m_prev = m_new

            out_tile = out_tile / l_tile.unsqueeze(-1)
            out[:, i*Bq:(i+1)*Bq, :] = out_tile
            lse_tile = m_prev + torch.log(l_tile)
            L_out[:, i*Bq:(i+1)*Bq] = lse_tile

        ctx.save_for_backward(Q, K, V, L_out, out)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L_out, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L_out, is_causal, 1.0 / (Q.shape[-1] ** 0.5))
        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_block_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Out_block_ptr = tl.make_block_ptr(
        Out_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_block_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_block_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q_tile = tl.load(Q_block_ptr)
    
    # 累加器必须是 fp32
    out_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_prev = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    l_tile = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    
    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        k_tile = tl.load(K_block_ptr)
        v_tile = tl.load(V_block_ptr)
        
        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale

        if is_causal:
            offs_n = k_start + tl.arange(0, K_TILE_SIZE)
            offs_m = query_block_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = offs_m[:, None] >= offs_n[None, :]
            s_tile = tl.where(mask, s_tile, -float('inf'))

        m_ij = tl.max(s_tile, axis=1)
        m_new = tl.maximum(m_prev, m_ij)

        p_tile = tl.exp(s_tile - m_new[:, None])
        l_tile_new = tl.exp(m_prev - m_new) * l_tile + tl.sum(p_tile, axis=1)
        
        alpha = tl.exp(m_prev - m_new)
        out_tile = alpha[:, None] * out_tile
        
        p_tile_cast = p_tile.to(v_tile.dtype)
        out_tile = tl.dot(p_tile_cast, v_tile, acc=out_tile)

        l_tile = l_tile_new
        m_prev = m_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    out_tile = out_tile / l_tile[:, None]
    
    # 转换回输入数据的类型 (bf16/fp16) 再存储
    out_tile_cast = out_tile.to(q_tile.dtype)
    tl.store(Out_block_ptr, out_tile_cast)
    
    # LSE is always float32
    lse_tile = m_prev + tl.log(l_tile)
    tl.store(L_block_ptr, lse_tile)
    

@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    kv_block_idx=tl.program_id(0)
    batch_idx=tl.program_id(1)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(kv_block_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(kv_block_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_idx * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(kv_block_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_idx * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(kv_block_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Q/O/dO/L: 从头遍历（offset=0）
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    k_tile = tl.load(K_block_ptr)
    v_tile = tl.load(V_block_ptr)

    dk_tile=tl.zeros((K_TILE_SIZE,D), dtype=tl.float32)
    dv_tile=tl.zeros((K_TILE_SIZE,D), dtype=tl.float32)

    for start_q in range(0, N_QUERIES, Q_TILE_SIZE):
        q_tile = tl.load(Q_block_ptr)
        o_tile = tl.load(O_block_ptr)
        do_tile = tl.load(dO_block_ptr)
        l_tile = tl.load(L_block_ptr)

        d_vec = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)

        # S = Q K^T / sqrt(d)
        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale

        if is_causal:
            offs_m = start_q + tl.arange(0, Q_TILE_SIZE)
            offs_n = kv_block_idx*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = offs_m[:, None] >= offs_n[None, :]
            s_tile = tl.where(mask, s_tile, -float('inf'))

        p_tile = tl.exp(s_tile - l_tile[:, None])
        p_tile_cast = p_tile.to(do_tile.dtype)

        dv_tile += tl.dot(tl.trans(p_tile_cast), do_tile)


        dp_tile = tl.dot(do_tile, tl.trans(v_tile))
        ds_tile = p_tile * (dp_tile - d_vec[:, None]) * scale

        # dQ 贡献（必须 atomic）
        ds_tile_cast = ds_tile.to(q_tile.dtype)
        dq_contrib = tl.dot(ds_tile_cast, k_tile)
        offs_q = start_q + tl.arange(0, Q_TILE_SIZE)
        offs_d = tl.arange(0, D)
        dq_ptrs = (
            dQ_ptr
            + batch_idx * stride_dqb
            + offs_q[:, None] * stride_dqq
            + offs_d[None, :] * stride_dqd
        )
        tl.atomic_add(dq_ptrs, dq_contrib)
       
        # dK += dS^T Q
        dk_tile += tl.dot(tl.trans(ds_tile_cast), q_tile)

        # advance Q/O/dO/L
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dk_tile)
    tl.store(dV_block_ptr, dv_tile)
    return



class FlashAttentionTT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq, Bk = 32, 32 # Kernel block sizes must match
        N, L, D = Q.shape
        assert L % Bq == 0 and L % Bk == 0
        Tq = L // Bq
        Tk = L // Bk

        out = torch.empty_like(Q)
        L_out = torch.empty((N, L), device=Q.device, dtype=torch.float32)

        scale = 1.0 / (D ** 0.5)

        grid = (Tq, N)

        flash_fwd_kernel[grid](
            Q, K, V, out, L_out,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            L_out.stride(0), L_out.stride(1),
            L, 
            L,
            scale,
            is_causal,
            D,
            Bq,
            Bk,
        )

        ctx.save_for_backward(Q, K, V, L_out, out)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L_out, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        # 用 fp32 缓冲区避免 atomic_add 类型问题
        dQ_fp32 = torch.zeros_like(Q, dtype=torch.float32)
        dK_fp32 = torch.zeros_like(K, dtype=torch.float32)
        dV_fp32 = torch.zeros_like(V, dtype=torch.float32)

        flash_bwd_kernel_args = (
            Q, K, V, O, dO, L_out,
            dQ_fp32, dK_fp32, dV_fp32,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L_out.stride(0), L_out.stride(1),
            dQ_fp32.stride(0), dQ_fp32.stride(1), dQ_fp32.stride(2),
            dK_fp32.stride(0), dK_fp32.stride(1), dK_fp32.stride(2),
            dV_fp32.stride(0), dV_fp32.stride(1), dV_fp32.stride(2),
            Q.shape[1],
            K.shape[1],
            1.0 / (Q.shape[-1] ** 0.5),
            is_causal,
            Q.shape[-1],
            32,
            32,
        )
        grid = (K.shape[1] // 32, Q.shape[0])
        flash_bwd_kernel[grid](*flash_bwd_kernel_args)
        return dQ_fp32.to(Q.dtype), dK_fp32.to(K.dtype), dV_fp32.to(V.dtype), None
