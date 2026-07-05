import os
import gc
import copy
from typing import List, Dict, Optional

import torch
import torch.distributed as dist
from transformers.modeling_utils import unwrap_model 
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
# from deepspeed.utils import safe_get_full_grad, safe_set_full_grad
from accelerate.state import AcceleratorState, PartialState
import deepspeed

from tqdm import tqdm
from .base import BaseCLearner
import sys
sys.path.append("/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM")


"""
LoTALearner
===========

- 针对 ZeRO-3 大模型的子矩阵级冻结方案；
- 训练前随机/自定义生成稀疏掩码，按任务序列逐渐加深稀疏；
- 反向传播时利用 safe_get_full_grad / safe_set_full_grad
  将被掩码位置的全梯度安全置零并写回；
- 支持 weight_decay 保护与 hook 撤销。
- sparsity_ratio指的是在训练中不参与运算的参数比例
"""
local_rank = None

def rank0_print(*args):
    if dist.get_rank() == 0:
        print(*args)

def rank0_log(msg: str):
    """
    Print & logging.info only on rank‑0, with step prefix.
    """
    if dist.get_rank() == 0:
        print(msg, flush=True)
        logging.info(msg)
def release_memory():
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats()


def memory_stats():
    rank0_print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
    rank0_print(f"memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2}")


def save_mask(mask: Dict[str, torch.Tensor], file_path: str):
    """
    Save the mask to a file.

    Parameters:
    - mask (dict): The mask to save.
    - file_path (str): The file path where the mask will be saved.
    """
    torch.save(mask, file_path)
    rank0_print(f"[LoTA] Mask saved to {file_path}")

import os, torch, logging
import torch.distributed as dist
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import pdb

def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Gather a full parameter tensor under ZeRO-3.
    Returns a **CPU** tensor detached from graph.
    """
    rank = dist.get_rank()
    
    if hasattr(param, "ds_id"):  # ZeRO shard
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logging.warning(f"{name}: param.ds_status == NOT_AVAILABLE")
        # shard 参数
        if rank == 0:
            with zero.GatheredParameters([param], modifier_rank=None):
                full = param.data.detach().cpu().clone()
            return full
        else:
            return None
    else:  # 已在当前进程
        return param.detach().cpu().clone() if rank == 0 else None
    

def get_global_sparsity_masks_zero3(
    model_params: Dict[str, torch.Tensor],      # 仅 rank-0 为 {name: full-tensor}，其余 rank 传入 {}
    sparsity_ratios: List[float],
    save_path: str,
    only_update_prune: bool = False,  #如果设为True则代表全部冻结
    bottom_k: bool = False, # True = 冻结最小 k；False = 冻结最大 k
) -> Dict[float, Dict[str, torch.Tensor]]:
    
    """
    仅用 model_params(已在 rank-0 收集的完整参数)计算全局阈值并生成稀疏掩码；
    其他 rank 不再重复通信大张量，避免 OOM:
    生成一个与参数矩阵相同形状,但是bool的矩阵,其True对应梯度裁剪的部分,False对应保留梯度的部分
    """
    rank0_print("开始计算mask矩阵")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_rank0 = rank == 0
    
    if is_rank0:
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)

    # ---------- 1) rank-0 统计绝对值 ----------

    abs_list = [torch.abs(t).view(-1) for  t in model_params.values()]
    all_abs  = torch.cat(abs_list) if abs_list else torch.empty(0)
    total_num  = all_abs.numel()
    # total_num_tensor = torch.tensor([total_num], dtype=torch.long, device='cpu')
    # dist.broadcast(total_num_tensor, src=0)
    # total_num = total_num.item()

    masks_all: Dict[float, Dict[str, torch.Tensor]] = {}
  
    for ratio in sparsity_ratios:
        # ---------- 2) rank‑0 计算阈值 ----------
        thresh_path = os.path.join(dirname,f'thresh_{ratio}.pt')
        k = int((1 - ratio) * total_num)
        if os.path.exists(thresh_path):
            thresh = torch.load(thresh_path)
        else:
            if k <= 0 or total_num == 0:             # 全剪 or 全保留的极端情况
                thresh = torch.tensor(float('inf' if bottom_k else '-inf'))
            else:
                thresh = torch.topk(all_abs, k, largest=not bottom_k).values[-1]   #取前k个最大值中的最小值
                torch.save(thresh, thresh_path)
        thresh = thresh.item()

        # ---------- 3) 各 rank 本地生成掩码 ----------
        mask_dict, kept_num = {}, 0
        
        pbar = tqdm(model_params.items(),
                total=len(model_params),
                ncols=80,
                disable=not is_rank0,
                desc=f"mask r={ratio:.2f}")
        
        for name, param in pbar:
            if only_update_prune:
                mask_dict[name] = torch.ones_like(param, dtype=torch.bool)
                kept_num += param.numel()
                
            if param is None:
                continue
            if bottom_k:
                param_mask = torch.abs(param) > thresh   # 保留较大权重
            else:
                param_mask = torch.abs(param) < thresh   # 比thresh小的值赋予True，比thresh大的值赋予False，True代表被冻结
                
            mask_dict[name] = param_mask.bool()
            kept_num += param_mask.sum().item()


        masks_all[ratio] = mask_dict
        pruned_pct = 100 * (total_num - kept_num) / total_num
        if is_rank0:
            logging.info(
                f"[LoTA-Global] ratio={ratio:.2f} | {pruned_pct:.2f}% pruned, "
                f"{100 - pruned_pct:.2f}% kept"
            )
    return masks_all
def get_named_parameters_list(model,pruning_fn,is_rank0: bool,)-> Dict[str, torch.Tensor]:
    """
    仅 rank-0：收集所有需要剪枝的权重绝对值，返回 {name: tensor.abs()}。
    其他 rank 返回空字典。
    """
    params_dict: Dict[str, torch.Tensor] = {}
    
    if is_rank0:
        for n, p in model.named_parameters():
            if (
                p.requires_grad
                and "weight" in n
                and pruning_fn_mm_filtered(n,pruning_fn)
            ):
                full_param = maybe_zero_3(p, name=n) # CPU tensor
 
                params_dict[n] = full_param
    else:
        params_dict = None

    
    return params_dict

def save_trainable_checkpoint(model, ckpt_path: str):
    """
    仅保存 requires_grad=True 的权重。dtype 与训练中的保持一致。
    只让 rank‑0；其他 rank 等待 barrier。
    """
    if dist.get_rank() == 0:
        trainable_state = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_state[n] = maybe_zero_3(p, name=n)  # CPU tensor, 原 dtype
        torch.save(trainable_state, ckpt_path)
        print(f"[LoTA‑Base] saved {len(trainable_state)} trainable params → {ckpt_path}")
        del trainable_state
        gc.collect()
    dist.barrier()
def restore_trainable_params(model, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")

    for n, p in model.named_parameters():
        if not p.requires_grad or n not in state:
            continue

        # 判断是否 ZeRO‑3 shard
        if hasattr(p, "ds_id"):        # 只有 ZeRO‑3 才会带 ds_id/ds_status
            with zero.GatheredParameters(p, modifier_rank=None):
                if getattr(p, "ds_status", None) == ZeroParamStatus.NOT_AVAILABLE:
                    continue           # 该 shard 不在当前 rank，跳过
                src = state[n].to(dtype=p.dtype, device=p.device, non_blocking=True)
                p.data.copy_(src)
        else:
            # 普通参数或 ZeRO‑2：直接覆盖
            src = state[n].to(dtype=p.dtype, device=p.device, non_blocking=True)
            p.data.copy_(src)

    del state
    torch.cuda.empty_cache()
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"[LoTA‑Base] parameters restored from {ckpt_path}")


def pruning_fn_mm_filtered(name: str,multimodal_keywords) -> bool:
    """
    判断参数名是否不属于多模态模块：
    """
    return not any(kw in name for kw in multimodal_keywords)

def compute_delta_params(
    pre_params_dict: Dict[str, torch.Tensor],
    ft_params_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    计算微调前后的参数差值（delta = ft - pre）
    参数应位于 CPU 上，结构为 {param_name: tensor}
    """
    delta_params_dict = {}
    for name in ft_params_dict:
        if name not in pre_params_dict:
            raise KeyError(f"Parameter {name} not found in pre-trained model.")
        
        # 差值计算（注意：确保两个 tensor 尺寸一致）
        delta = ft_params_dict[name] - pre_params_dict[name]
        delta_params_dict[name] = delta
    
    return delta_params_dict
    
class LoTALearner(BaseCLearner):
    """
    LoTA = **Lo**cal **T**ensor-wise **A**ggregate Sparsity Learner

    - 在每个任务开始前生成 / 载入一个随机稀疏掩码 (BoolTensor)；
    - 通过 `param.register_hook` 将梯度中被掩码位置置零，实现 *“子矩阵级冻结”*；
    - 支持多任务：`self.sparsity_ratios[task_id-1]` 决定当前任务的 sparsity。
    """
    def __init__(self,trainer_cls,sparsity_ratios: List[float],pruning_fn,training_args,start_task_id,model):
        super().__init__()
        self.model = model
        self.training_args = copy.deepcopy(training_args)
        self.output_dir = self.training_args.output_dir
        self.start_task_id = start_task_id
        self.train_cl_learner = BaseCLearner()
        
        self.sparsity_ratios = sparsity_ratios
        self.pruning_fn = pruning_fn
        self.num_data_for_mask = 200  #for test
        self.mask_dir = os.path.join(training_args.output_dir, "lota_masks")
        os.makedirs(self.mask_dir, exist_ok=True)
        
        # 保存 hook handle，方便 after_train 解除
        self._handles: List[torch.utils.hooks.RemovableHandle] = []    
        self.trainer_cls = trainer_cls
        
    # public hooks  
    def before_train(
        self,
        task_id: int,
        model: torch.nn.Module,
        tokenizer=None,
        train_dataset=None,  
        data_collator=None,
        **unused,
    ):
        """
        在 LLaVATrainer 训练该任务之前调用：
        1. 生成 / 加载掩码；
        2. 为对应参数注册梯度钩子，屏蔽掩码位置的梯度。
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_rank0 = rank == 0
        idx = min(task_id - 1, len(self.sparsity_ratios) - 1)
        cur_ratio = self.sparsity_ratios[idx]
        #每个任务掩码的位置
        mask_path_dir = os.path.join(self.mask_dir,str(task_id))
        os.makedirs(mask_path_dir,exist_ok=True)
        mask_path = os.path.join(mask_path_dir, f"{cur_ratio:.2f}_mask.pt")

        if not task_id > self.start_task_id:
            rank0_log(f"[LoTA] ➡️  Enter before_train | task_id={task_id}")
            return
        if os.path.exists(mask_path):
            mask_dict = torch.load(mask_path)
        else:
            #保存当前模型参数
            base_ckpt_path = os.path.join(self.training_args.output_dir, f"base_fp32_{task_id}.pt")
            if not os.path.isfile(base_ckpt_path):
                if is_rank0:
                    logging.info("Saving base checkpoint before CL starts ...")
                save_trainable_checkpoint(model, base_ckpt_path)
                rank0_log("[LoTA] ✔ Base ckpt ready")
            #1) 挑选当前任务要用的稀疏率

            idx = min(task_id - 1, len(self.sparsity_ratios) - 1)
            cur_ratio = self.sparsity_ratios[idx]
            #获得pre_model_params
            if dist.get_rank() == 0:
                logging.info(f"[LoTA]  rank-0 记录原模型需要被掩码的参数 (ratio={cur_ratio:.2f})，开始提取需要掩码的参数")
                pre_params_dict = get_named_parameters_list(model,self.pruning_fn,is_rank0)
                rank0_log(f"[LoTA] ✔ Collected pre_params_dict ({len(pre_params_dict)} tensors)")
                
            dist.barrier()  # 保证所有进程同步开始
            self.training_args.output_dir = os.path.join(self.output_dir, f"{str(task_id)}_pre", "pre_ft")
            if len(train_dataset) > self.num_data_for_mask:
                interval = int(len(train_dataset) // self.num_data_for_mask)
                train_dataset.list_data_dict = train_dataset.list_data_dict[::interval]
                train_dataset_nums = len(train_dataset.list_data_dict)
            if is_rank0:
                logging.info(f"开始提前训练以计算掩码参数,训练的数据量是{train_dataset_nums})")
            release_memory()

            trainer = self.trainer_cls(cl_learner=self.train_cl_learner,
                                        train_dataset=train_dataset,
                                        data_collator=data_collator,
                                        args=self.training_args,
                                        model=model,
                                        tokenizer=tokenizer,)
            
            
            trainer.train()
            
            #获得ft_model_params
            dist.barrier()  # 保证所有进程同步开始
            if dist.get_rank() == 0:
                logging.info(f"提前训练完毕，开始提取需要掩码的参数")
                ft_params_dict = get_named_parameters_list(trainer.model,self.pruning_fn,is_rank0)
                delta_params_dict = compute_delta_params(ft_params_dict,pre_params_dict)
    
                del ft_params_dict
                del pre_params_dict
            rank0_log("[LoTA] ✔ Mask dict saved/loaded")
            del trainer
            # try:
            #     AcceleratorState().deactivate()
            # except AttributeError:
            #     pass
            # PartialState._shared_state = {}
            # if hasattr(deepspeed.runtime, "engine"):
            #     deepspeed.runtime.engine.DEEPSPEED_ENGINE = None
            dist.barrier()
            release_memory()
            
            #2) 生成 / 载入掩码（ZeRO-3 友好）

            dist.barrier() 
            # if dist.get_rank() == 0 and not os.path.exists(mask_path):
            if dist.get_rank() == 0:
                logging.info(f"[LoTA]  rank-0 生成稀疏掩码 (ratio={cur_ratio:.2f})")
                all_masks = get_global_sparsity_masks_zero3(
                    model_params=delta_params_dict,
                    sparsity_ratios=[cur_ratio],
                    save_path=mask_path,   
                )
                mask_dict = all_masks[cur_ratio]           #  rank0 拿到mask字典
                torch.save(mask_dict,mask_path)
            dist.barrier() 
            if not dist.get_rank() == 0:
                mask_dict = torch.load(mask_path)
            
            # === 3) 梯度回滚到 base ckpt =========================
            rank0_log("[LoTA] ➡️  Restoring trainable params from base ckpt")
            
            restore_trainable_params(model, base_ckpt_path)
        dist.barrier()
        # === 4) 为每个需要稀疏的参数注册梯度 Hook =========================
        self._register_gradient_mask(model, mask_dict)
        rank0_log(f"[LoTA] ✔ Hooks registered | sparsity={cur_ratio:.2f}")
        if dist.get_rank() == 0:
            logging.info(f"[LoTA] Task-{task_id} 掩码注册完毕；稀疏率 {cur_ratio:.2f}")
        rank0_print(f"[LoTA] Task-{task_id} 掩码注册完毕；稀疏率 {cur_ratio:.2f}")
        
        
    def after_train(self, *args, **kwargs):
        """
        训练结束后，撤销所有 hook，防止泄露到下个任务。
        """
        for h in self._handles:
            h.remove()
        self._handles.clear()
        
    def _register_gradient_mask(self,
                                model: torch.nn.Module,
                                mask_dict: Dict[str, torch.Tensor]) -> None:
        """
        两阶段冻结：
        (1) register_hook        → 置 0 本地 shard 梯度
        # (2) register_full_bw_hook→ 置 0 完整梯度后写回
        """
        for name, p in model.named_parameters():
            if name not in mask_dict:
                continue
            if not p.requires_grad:          # 已冻结的不挂 hook
                continue

            mask_cpu = mask_dict[name]       # BoolTensor 全尺寸
            mask_gpu = mask_cpu.to(p.device, non_blocking=True)

            # ---------- (1) shard 级 hook ----------
            def _local_grad_hook(grad, _m_gpu=mask_gpu, _p=p):
                if grad is None:
                    return None
                # ZeRO-3 shard 情况
                with torch.no_grad():
                    if hasattr(_p, "ds_tensor_slice"):
                        sl = _p.ds_tensor_slice           # 当前 rank shard
                        assert grad.numel()== mask_gpu[sl].numel()
                        grad[_m_gpu[sl]] = 0             # True → 置 0
                    else:
                        grad[_m_gpu] = 0
                    return grad
            self._handles.append(p.register_hook(_local_grad_hook))

        if getattr(self.training_args, "weight_decay", 0.0) > 0:
            def _keep_masked_weight(module, *_,
                                    _md=mask_dict):
                for n, w in module.named_parameters(recurse=False):
                    if n in _md:
                        w.data[_md[n].to(w.device, non_blocking=True)] = \
                            w.data[_md[n].to(w.device)]
            self._handles.append(model.register_forward_pre_hook(_keep_masked_weight))
            
   
      