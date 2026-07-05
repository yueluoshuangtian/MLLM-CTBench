# moelora_learner_zero2.py
import os
import json
import csv
from typing import Optional, Dict, Any
import ipdb
import torch
import torch.distributed as dist
from .base import BaseCLearner


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def _rank0(training_args) -> bool:
    r = getattr(training_args, "local_rank", 0)
    return (r == 0 or r == -1)

class moeloraLearner(BaseCLearner):
    """
    ZeRO-2 友好版:
      - 训练中累计逐层/逐专家的 count、weight_sum、total_samples、total_selections；
      - 支持按步/按 epoch 快照；
      - 在导出前用 all_reduce(sum) 做全局归并；
      - 提供 enable_record() 门控,避免梯度检查点重复计数。
    """
    def __init__(
        self,
        training_args,
        start_task_id,
        model,
        snapshot_every_n_steps: int = 0,
        snapshot_on_epoch_end: bool = True,
        
    ):
        self.training_args = training_args
        self.start_task_id = start_task_id if start_task_id>0 else 1
        self.adapter_name = getattr(training_args, "adapter_name", "coin_moe")

        self.per_layer_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._registered_layers = set()

        self.snapshot_every_n_steps = int(snapshot_every_n_steps)
        self.snapshot_on_epoch_end = bool(snapshot_on_epoch_end)
        self._last_snapshot_step = -1

        self.out_root = training_args.output_dir

        # —— 门控,配合梯度检查点去重 —— #
        self._record_enabled = True  # 默认开；Trainer 每步可显式开/关
    # ---------- 公共门控 API ----------
    def enable_record(self, flag: bool):
        self._record_enabled = bool(flag)

    # ---------- 判定 + 载荷还原 ----------
    @staticmethod
    def _is_coin_moe_linear(m):
        need = ("add_one_expert", "_current_E", "freeze_expert", "lora_router", "usage_recorder")
        return all(hasattr(m, t) for t in need)

    @staticmethod
    def _payload_to_full_router(payload: Any, device=None) -> Optional[torch.Tensor]:
        if isinstance(payload, torch.Tensor):
            return payload
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("router_full", None), torch.Tensor):
            return payload["router_full"]
        router = payload.get("router", None)
        active_idx = payload.get("active_idx", None)
        E = int(payload.get("E", 0)) if payload.get("E", None) is not None else 0
        if isinstance(router, torch.Tensor) and isinstance(active_idx, torch.Tensor) and E > 0:
            B, _ = router.shape
            dev = device or router.device
            full = torch.zeros(B, E, device=dev, dtype=router.dtype)
            full.index_copy_(dim=1, index=active_idx.to(dev), source=router.to(dev))
            return full
        if isinstance(router, torch.Tensor):
            return router
        return None

    # ---------- 统计累加 ----------
    def _ensure_layer_bucket(self, name: str, E: int):
        if name in self.per_layer_stats:
            curE = int(self.per_layer_stats[name]["E"])
            if E > curE:
                for key in ("count", "weight_sum"):
                    old = self.per_layer_stats[name][key]
                    new = torch.zeros(E, dtype=old.dtype)
                    new[:curE] = old
                    self.per_layer_stats[name][key] = new
                self.per_layer_stats[name]["E"] = E          # ← 直接存 int
            return
        self.per_layer_stats[name] = {
            "count": torch.zeros(E, dtype=torch.long),
            "weight_sum": torch.zeros(E, dtype=torch.float32),
            "total_samples": torch.zeros(1, dtype=torch.long),
            "total_selections": torch.zeros(1, dtype=torch.long),
            "E": E,                                          # ← 直接存 int
        }

    def _accumulate_batch(self, layer_name: str, router_full_cpu: torch.Tensor):
        B, e = router_full_cpu.shape
        st = self.per_layer_stats[layer_name]
        st["total_samples"] += B
        sel = (router_full_cpu > 0).to(torch.long)
        st["total_selections"] += int(sel.sum().item())
        with torch.no_grad():
            st["count"][:e] += sel.sum(dim=0).to(torch.long)
            st["weight_sum"][:e] += router_full_cpu.sum(dim=0).to(torch.float32)

    # ---------- 分布式归并(ZeRO-2:sum 即可) ----------
    def _all_reduce_stats_inplace(self):
        """
        分布式环境下原地归并所有进程的统计数据(修改原张量)
        核心逻辑:CPU统计张量 → 转移到GPU → 多卡求和 → 转回CPU并恢复原类型
        适配NCCL后端(GPU间通信),避免CPU张量直接归并的兼容性问题
        """
        # 1. 非分布式环境无需归并,直接返回
        if not _is_dist():
            return
        
        # 2. 无CUDA环境时不执行(NCCL后端依赖GPU,CPU分布式通常用Gloo,此处简化处理)
        if not torch.cuda.is_available():
            # 没有 CUDA 就别做 NCCL all_reduce,避免后端/CPU 设备不匹配
            return

        # 3. 获取当前进程的GPU设备(确保张量在正确的GPU上进行归并)
        # torch.cuda.current_device() 返回当前进程使用的GPU索引(如0,1,2...)
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")

        # 4. 遍历所有层的统计数据,逐张量归并
        for st in self.per_layer_stats.values():
            # 每个"st"是一个层的统计字典(含count、weight_sum等键)
            for key in ("count", "weight_sum", "total_samples", "total_selections"):
                t = st[key]  # 当前统计项的张量(存储在CPU上,避免占用GPU内存)

                # 5. 将CPU张量转移到当前GPU,并转为float32类型
                # 原因:NCCL对float32支持更稳定,且求和操作在浮点型下精度更高
                # 显式指定device和dtype为关键字参数,避免位置参数歧义
                tmp = t.to(device=dev, dtype=torch.float32)

                # 6. 执行分布式归并:所有进程的tmp张量求和(in-place操作)
                # dist.all_reduce默认使用当前进程组,op=SUM指定求和操作
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)

                # 7. 将归并结果转回CPU,并恢复原始数据类型
                if t.dtype.is_floating_point:
                    # 原始为浮点型(如weight_sum):直接转回原 dtype
                    t.copy_(tmp.to(device="cpu", dtype=t.dtype))
                else:
                    # 原始为整型(如count、total_samples):先四舍五入再转回原 dtype
                    # 避免浮点求和后的小数部分导致整型转换误差(如2.1→2,2.9→3)
                    t.copy_(tmp.round().to(device="cpu", dtype=t.dtype))


    # ---------- 导出 ----------
    def _dump_stats_files(self, out_dir: str, tag: str):
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"moelora_router_stats_{tag}.csv")
        json_path = os.path.join(out_dir, f"moelora_router_stats_{tag}.json")

        export = {}
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([
                "layer","expert_idx",
                "count","weight_sum",
                "freq_in_samples",              # count / total_samples
                "avg_weight_when_selected",     # weight_sum / count
                "avg_select_ratio",             # count / total_selections
                "avg_weight_over_selections",   # weight_sum / total_selections
                "total_samples","total_selections"
            ])
            for layer, st in self.per_layer_stats.items():
                count = st["count"]
                weight_sum = st["weight_sum"]
                total_samples = int(st["total_samples"].item())
                total_selections = int(st["total_selections"].item())
                E = int(st["E"].item() if isinstance(st["E"], torch.Tensor) else st["E"])

                if total_samples > 0:
                    freq_in_samples = count.to(torch.float32) / float(total_samples)
                else:
                    freq_in_samples = torch.zeros_like(weight_sum, dtype=torch.float32)

                avg_weight_when_selected = torch.zeros_like(weight_sum, dtype=torch.float32)
                nz = (count > 0)
                if nz.any():
                    avg_weight_when_selected[nz] = weight_sum[nz] / count[nz].to(torch.float32)

                if total_selections > 0:
                    inv_sel = 1.0 / float(total_selections)
                    avg_select_ratio = count.to(torch.float32) * inv_sel
                    avg_weight_over_selections = weight_sum * inv_sel
                else:
                    avg_select_ratio = torch.zeros_like(weight_sum, dtype=torch.float32)
                    avg_weight_over_selections = torch.zeros_like(weight_sum, dtype=torch.float32)
                
                export[layer] = {
                    "E": E,
                    "total_samples": total_samples,
                    "total_selections": total_selections,
                    "count": count.tolist(),
                    "weight_sum": weight_sum.tolist(),
                    "freq_in_samples": freq_in_samples.tolist(),
                    "avg_weight_when_selected": avg_weight_when_selected.tolist(),
                    "avg_select_ratio": avg_select_ratio.tolist(),
                    "avg_weight_over_selections": avg_weight_over_selections.tolist(),
                }
                for i in range(E):
                    wr.writerow([
                        layer, i,
                        int(count[i].item()),
                        float(weight_sum[i].item()),
                        float(freq_in_samples[i].item()) if total_samples>0 else 0.0,
                        float(avg_weight_when_selected[i].item()),
                        float(avg_select_ratio[i].item()) if total_selections>0 else 0.0,
                        float(avg_weight_over_selections[i].item()) if total_selections>0 else 0.0,
                        total_samples, total_selections
                    ])
        with open(json_path, "w") as jf:
            json.dump(export, jf, ensure_ascii=False, indent=2)

    def _dump_expert_weights(self, model, out_dir: str, tag: str):
        # ZeRO-2:参数是全量复制,rank0 直接取
        os.makedirs(out_dir, exist_ok=True)
        dump = {}
        for layer, module in model.named_modules():
            if not self._is_coin_moe_linear(module):
                continue
            try:
                E = int(module._current_E())
            except Exception:
                continue
            try:
                A_list = module.lora_A[module.active_adapter].loraA
                B_list = module.lora_B[module.active_adapter].loraB
                R_list = module.lora_router[module.active_adapter]
            except Exception:
                continue
            pack = {"experts": {}}
            for i in range(E):
                try:
                    A_w = A_list[i].mlp.weight.detach().cpu().clone()
                    B_w = B_list[i].mlp.weight.detach().cpu().clone()
                    R_w = R_list[i].weight.detach().cpu().clone()
                except Exception:
                    continue
                pack["experts"][i] = {"A": A_w, "B": B_w, "router": R_w}
            dump[layer] = pack
        torch.save(dump, os.path.join(out_dir, f"moelora_expert_weights_{tag}.pt"))
        # 元数据
        meta = {}
        for layer, pack in dump.items():
            em = {}
            for i, blobs in pack["experts"].items():
                em[int(i)] = {
                    "A_shape": tuple(blobs["A"].shape),
                    "B_shape": tuple(blobs["B"].shape),
                    "router_shape": tuple(blobs["router"].shape),
                }
            meta[layer] = em
        with open(os.path.join(out_dir, f"moelora_expert_weights_{tag}.meta.json"), "w") as jf:
            json.dump(meta, jf, ensure_ascii=False, indent=2)

    # ---------- Hooks ----------
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        for name, module in model.named_modules():
            if not self._is_coin_moe_linear(module) or name in self._registered_layers:
                continue

            # 建立一份层 bucket(省略:你的 _ensure_layer_bucket 实现)
            try:
                E = int(module._current_E())
            except Exception:
                continue
            self._ensure_layer_bucket(name, E)

            # —— 关键:recorder 闭包,入口第一行检查门控 —— #
            def make_recorder(layer_name, module_ref):
                def _recorder(payload):
                    if not self._record_enabled:
                        return

                    router_full = self._payload_to_full_router(payload)
                    if router_full is None:
                        if isinstance(payload, torch.Tensor):
                            try:
                                active_idx = module_ref._active_indices()
                                E_full = int(module_ref._current_E())
                                B = payload.shape[0]
                                # 直接在 CPU 上重建,减少显存占用
                                full = torch.zeros(B, E_full, dtype=payload.dtype)
                                full.index_copy_(dim=1, index=active_idx.cpu(), source=payload.detach().cpu())
                                router_full = full
                            except Exception:
                                return
                        else:
                            return

                    # —— 统一转 CPU 再累计 —— #
                    router_full_cpu = router_full.detach().to("cpu")
                    self._ensure_layer_bucket(layer_name, router_full_cpu.shape[1])
                    self._accumulate_batch(layer_name, router_full_cpu)
                return _recorder

            module.usage_recorder = make_recorder(name, module)
            self._registered_layers.add(name)

        self._last_snapshot_step = -1
        # 默认允许记录
        self._record_enabled = True

    def after_train(self, task_id, model,data_modules, **kwargs):
        # 解绑 recorder
        for _, module in model.named_modules():
            if self._is_coin_moe_linear(module) and getattr(module, "usage_recorder", None) is not None:
                module.usage_recorder = None

        # —— 导出任务总值(先做 all_reduce 合并) —— #
        self._all_reduce_stats_inplace()
        if _rank0(self.training_args):
            out_dir = getattr(self.training_args, "output_dir", self.out_root)
            tag = f"task{task_id}_final"
            self._dump_stats_files(out_dir, tag)
            self._dump_expert_weights(model, out_dir, tag)

        # 清理
        self.per_layer_stats.clear()
        self._registered_layers.clear()

    # —— 供 Trainer 每步结束时调用:阶段快照(合并后再写) —— #
    def on_step_end(self, global_step: int, epoch: Optional[int] = None, out_dir: Optional[str] = None):
        # 阶段快照频率控制
        if self.snapshot_every_n_steps <= 0:
            return
        if self._last_snapshot_step >= 0 and (global_step - self._last_snapshot_step) < self.snapshot_every_n_steps:
            return
        
        # 合并后 rank0 落盘
        self._all_reduce_stats_inplace()
        if _rank0(self.training_args):
            od = out_dir or getattr(self.training_args, "output_dir", self.out_root)
            snap_dir = os.path.join(od ,"snapshots")
            tag = f"step_{global_step}" if epoch is None else f"epoch_{epoch}_step_{global_step}"
            self._dump_stats_files(snap_dir, tag)
        self._last_snapshot_step = global_step

    # —— 供 Trainer 每个 epoch 结束时调用:快照 —— #
    def on_epoch_end(self, epoch: int, out_dir: Optional[str] = None):
        if not self.snapshot_on_epoch_end:
            return
        self._all_reduce_stats_inplace()
        if _rank0(self.training_args):
            od = out_dir or getattr(self.training_args, "output_dir", self.out_root)
            snap_dir = os.path.join(od, "snapshots")
            tag = f"epoch_{epoch}"
            self._dump_stats_files(snap_dir, tag)
