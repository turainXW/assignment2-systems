
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
    scale: float):
    # Placeholder for backward implementation
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    Bq,Bk=16,16
    N, Length, D = Q.shape
    assert Length % Bq == 0 and Length % Bk == 0
    Tq = Length // Bq
    Tk = Length // Bk
    Dvec=torch.sum(dO*O,dim=-1)

    for i in range(0, Tk):
        k_tile=K[:,i*Bk:(i+1)*Bk,:]
        v_tile=V[:,i*Bk:(i+1)*Bk,:]
        for j in range(0, Tq):
            q_tile=Q[:,j*Bq:(j+1)*Bq,:]
            o_tile=O[:,j*Bq:(j+1)*Bq,:]
            do_tile=dO[:,j*Bq:(j+1)*Bq,:]
            dq_tile=torch.zeros((N,Bq,D), device=Q.device, dtype=Q.dtype)
            dk_tile=torch.zeros((N,Bk,D), device=Q.device, dtype=K.dtype)
            dv_tile=torch.zeros((N,Bk,D), device=Q.device, dtype=V.dtype)
            l_tile=L[:,j*Bq:(j+1)*Bq]
            D_tile=Dvec[:,j*Bq:(j+1)*Bq]
            # The actual backward computation should be implemented here
            # This is a placeholder to illustrate the structure

            s_tile=torch.einsum('...bd,...kd->...bk',q_tile,k_tile)/(D**0.5)
            if is_causal:
                q_pos=torch.arange(j*Bq,(j+1)*Bq,device=Q.device)
                k_pos=torch.arange(i*Bk,(i+1)*Bk,device=Q.device)
                mask=(k_pos[None,:]<=q_pos[:,None])
                s_tile=s_tile.masked_fill(~mask[None,:,:],float('-inf'))

            p_tile=torch.exp(s_tile-l_tile.unsqueeze(-1))
            dv_tile+=torch.einsum('...qk,...qd->...kd',p_tile,do_tile)
            dp_tile=torch.einsum('...qd,...kd->...qk',do_tile,v_tile)
            ds_tile=p_tile*(dp_tile-D_tile.unsqueeze(-1))/(D**0.5)
            dq_tile+=torch.einsum('...qk,...kd->...qd',ds_tile,k_tile)
            dk_tile+=torch.einsum('...qk,...qd->...kd',ds_tile,q_tile)
            dK[:,i*Bk:(i+1)*Bk,:]+=dk_tile
            dV[:,i*Bk:(i+1)*Bk,:]+=dv_tile
            dQ[:,j*Bq:(j+1)*Bq,:]+=dq_tile
    return dQ,dK,dV



class FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq,Bk=16,16
        N, L, D = Q.shape
        assert L % Bq == 0 and L % Bk == 0
        Tq = L // Bq
        Tk = L // Bk

        out = torch.empty_like(Q)
        L_out = torch.empty(( N, L), device=Q.device, dtype=torch.float32)

        for i in range(0, Tq):
            q_tile=Q[:,i*Bq:(i+1)*Bq,:]
            out_tile = torch.zeros((N,Bq,D), device=Q.device, dtype=Q.dtype)
            m_prev = torch.full((N,Bq),float('-inf'), device=Q.device, dtype=torch.float32)
            l_tile = torch.zeros((N,Bq), device=Q.device, dtype=torch.float32)
            for j in range(0, Tk):
                k_tile=K[:,j*Bk:(j+1)*Bk,:]
                v_tile=V[:,j*Bk:(j+1)*Bk,:]

                s_tile=torch.einsum('...bd,...kd->...bk',q_tile,k_tile)/(D**0.5)

                if is_causal:
                    q_pos = torch.arange(i * Bq, (i + 1) * Bq, device=Q.device)  # (Bq,)
                    k_pos = torch.arange(j * Bk, (j + 1) * Bk, device=Q.device)  # (Bk,)
                    mask = (k_pos[None, :] <= q_pos[:, None])  # (Bq,Bk)
                    s_tile = s_tile.masked_fill(~mask[None, None, :, :], float('-inf'))

                m_ij=torch.max(s_tile,dim=-1).values
                m_new=torch.maximum(m_prev,m_ij)

                p_tile=torch.exp(s_tile - m_new.unsqueeze(-1))
                l_tile_new=torch.exp(m_prev-m_new)*l_tile+p_tile.sum(dim=-1)
                out_tile=torch.einsum('...bk,...kd->...bd',p_tile,v_tile)+torch.exp(m_prev-m_new).unsqueeze(-1)*out_tile

                l_tile=l_tile_new
                m_prev=m_new

            out_tile=out_tile/l_tile.unsqueeze(-1)
            out[:,i*Bq:(i+1)*Bq,:]=out_tile
            lse_tile = m_prev + torch.log(l_tile)  # <-- this is the usual "LSE"
            L_out[:, i * Bq:(i + 1) * Bq] = lse_tile  # <-- store this instead
        ctx.save_for_backward(Q, K, V, L_out,out)
        ctx.is_causal = is_causal
        return out
    def backward(ctx, dO):
        Q, K, V, L_out,O = ctx.saved_tensors
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
    batch_idx=tl.program_id(1)

    Q_block_ptr=tl.make_block_ptr(
        Q_ptr+batch_idx*stride_qb,
        shape=(N_QUERIES,D),
        strides=(stride_qq,stride_qd),
        offsets=(query_block_idx*Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE,D),
        order=(1,0),
    )

    K_block_ptr=tl.make_block_ptr(
        K_ptr+batch_idx*stride_kb,
        shape=(N_KEYS,D),
        strides=(stride_kk,stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE,D),
        order=(1,0),
    )

    V_block_ptr=tl.make_block_ptr(
        V_ptr+batch_idx*stride_vb,
        shape=(N_KEYS,D),
        strides=(stride_vk,stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE,D),
        order=(1,0),
    )

    Out_block_ptr=tl.make_block_ptr(
        Out_ptr+batch_idx*stride_ob,
        shape=(N_QUERIES,D),
        strides=(stride_oq,stride_od),
        offsets=(query_block_idx*Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE,D),
        order=(1,0),
    )
    L_block_ptr=tl.make_block_ptr(
        L_ptr+batch_idx*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_block_idx*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q_tile=tl.load(Q_block_ptr)
    out_tile=tl.zeros((Q_TILE_SIZE,D),dtype=tl.float32)
    m_prev=tl.full((Q_TILE_SIZE,),-float('inf'),dtype=tl.float32)
    l_tile=tl.zeros((Q_TILE_SIZE,),dtype=tl.float32)
    for k_start in range(0,N_KEYS,K_TILE_SIZE):
        k_tile=tl.load(K_block_ptr)
        v_tile=tl.load(V_block_ptr)
        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale

        if is_causal:
            offs_n = k_start + tl.arange(0, K_TILE_SIZE)
            offs_m = query_block_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            # 利用广播机制生成掩码矩阵 (Q_TILE_SIZE, K_TILE_SIZE)
            # 只有 i >= j 的位置是合法的
            mask = offs_m[:, None] >= offs_n[None, :]
            
            # 将不合法的位置设为负无穷
            s_tile = tl.where(mask, s_tile, -float('inf'))

        m_ij=tl.max(s_tile,axis=1)
        m_new=tl.maximum(m_prev,m_ij)

        p_tile=tl.exp(s_tile - m_new[:,None])
        l_tile_new=tl.exp(m_prev-m_new)*l_tile+tl.sum(p_tile,axis=1)
        alpha = tl.exp(m_prev - m_new)                      # fp32
        out_tile = alpha[:, None] * out_tile                # fp32
        p_tile_cast = p_tile.to(v_tile.dtype)
        out_tile = tl.dot(p_tile_cast, v_tile, acc=out_tile)     # acc= 保持 fp32 累积


        l_tile=l_tile_new
        m_prev=m_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    out_tile=out_tile/l_tile[:,None]
    tl.store(Out_block_ptr,out_tile)
    lse_tile = m_prev + tl.log(l_tile)
    tl.store(L_block_ptr, lse_tile)


            



    

class FlashAttentionTT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq,Bk=16,16
        N, L, D = Q.shape
        assert L % Bq == 0 and L % Bk == 0
        Tq = L // Bq
        Tk = L // Bk

        out = torch.empty_like(Q)
        L_out = torch.empty(( N, L), device=Q.device, dtype=torch.float32)

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

        ctx.save_for_backward(Q, K, V, L_out,out)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L_out,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L_out, is_causal, 1.0 / (Q.shape[-1] ** 0.5))
        return dQ, dK, dV, None

