
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
class DDP(torch.nn.Module):
    def __init__(self, module:torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.handles=[]

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data,src=0)

        for param in self.module.parameters():
            if param.requires_grad:  
                param.register_post_accumulate_grad_hook(self._make_hook(param))
    
    def _make_hook(self, param):

        def hook(*args):
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((param,handle))
        return hook

    def forward(self, *inputs, **kwargs):
        self.handles=[]
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for _, handle in self.handles:
            handle.wait()
        self.handles=[]

        with torch.no_grad():
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad.data /= self.world_size



class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        
        # 1. 初始广播
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # 2. 静态分桶
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets = [] # 存储格式: [{"params": [], "count": 0, "handle": None, "flat_grads": None}, ...]
        self.param_to_bucket_id = {}
        
        # 逆序获取所有需要梯度的参数
        params_list = [p for p in self.module.parameters() if p.requires_grad]
        reversed_params = list(reversed(params_list))
        
        current_bucket = []
        current_size = 0
        for p in reversed_params:
            p_size = p.numel() * p.element_size()
            if current_size + p_size > self.bucket_size_bytes and current_bucket:
                # 封存一个桶
                bucket_id = len(self.buckets)
                for param in current_bucket:
                    self.param_to_bucket_id[id(param)] = bucket_id
                self.buckets.append({"params": current_bucket, "count": 0, "handle": None})
                current_bucket = []
                current_size = 0
            
            current_bucket.append(p)
            current_size += p_size
            
        if current_bucket:
            bucket_id = len(self.buckets)
            for param in current_bucket:
                self.param_to_bucket_id[id(param)] = bucket_id
            self.buckets.append({"params": current_bucket, "count": 0, "handle": None})

        # 3. 注册钩子
        for p in params_list:
            p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param):
        def hook(*args):
            bucket_id = self.param_to_bucket_id[id(param)]
            bucket = self.buckets[bucket_id]
            bucket["count"] += 1
            
            # 当桶内所有梯度都准备好了
            if bucket["count"] == len(bucket["params"]):
                # 1. 提取所有梯度
                grads = [p.grad.data for p in bucket["params"]]
                # 2. 打包 (Flatten)
                bucket["flat_grads"] = _flatten_dense_tensors(grads)
                # 3. 异步 All-Reduce
                bucket["handle"] = dist.all_reduce(
                    bucket["flat_grads"], op=dist.ReduceOp.SUM, async_op=True
                )
        return hook

    def forward(self, *inputs, **kwargs):
        # 每一轮开始前，重置桶计数器和句柄
        for bucket in self.buckets:
            bucket["count"] = 0
            bucket["handle"] = None
            bucket["flat_grads"] = None
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        with torch.no_grad():
            for bucket in self.buckets:
                if bucket["handle"] is not None:
                    # 1. 等待通信完成
                    bucket["handle"].wait()
                    # 2. 求平均 (可以在 flat_grads 上直接操作，效率更高)
                    bucket["flat_grads"] /= self.world_size
                    # 3. 写回 (Unflatten): 将同步后的数据拷贝回 param.grad.data
                    synced_grads = _unflatten_dense_tensors(bucket["flat_grads"], [p.grad.data for p in bucket["params"]])
                    for original_grad, new_grad in zip([p.grad.data for p in bucket["params"]], synced_grads):
                        original_grad.copy_(new_grad)
                
                # 清理内存引用
                bucket["handle"] = None
                bucket["flat_grads"] = None

