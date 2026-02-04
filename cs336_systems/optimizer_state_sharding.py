import torch
from torch.optim import Optimizer
from typing import Type, Any, List, Optional
import torch.distributed as dist

class optimizer_state_sharding(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optim_kwargs = kwargs
        
        # 建议只使用一个内部优化器来管理所有 group
        self.internal_optim: Optional[Optimizer] = None

        # 这一步会触发 add_param_group
        super().__init__(params, kwargs)

    def step(self, closure=None, **kwargs):
        loss = None
        if closure is not None:
            loss = closure()
        
        # 只需要调用一次内部优化器的 step
        if self.internal_optim is not None:
            self.internal_optim.step()

        self._all_gather_params()
        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        all_params = list(param_group['params'])
        
        # 分片逻辑
        my_params = [p for i, p in enumerate(all_params) if i % self.world_size == self.rank]
        
        # 组装本地参数组配置
        local_group = {k: v for k, v in param_group.items() if k != 'params'}
        local_group['params'] = my_params

        # 初始化或添加组到内部优化器
        if self.internal_optim is None:
            self.internal_optim = self.optimizer_cls([local_group], **self.optim_kwargs)
        else:
            self.internal_optim.add_param_group(local_group)

        # 必须调用父类方法，确保 self.param_groups 被正确维护
        super().add_param_group(param_group)

    @torch.no_grad()
    def _all_gather_params(self):
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                owner_rank = i % self.world_size
                dist.broadcast(p, src=owner_rank)

    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)
        if self.internal_optim is not None:
            self.internal_optim.zero_grad(set_to_none=set_to_none)
