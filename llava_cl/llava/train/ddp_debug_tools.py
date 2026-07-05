# -*- coding: utf-8 -*-
import os, time
import torch
import torch.distributed as dist
from transformers import TrainerCallback

def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _rank():
    try:
        return torch.distributed.get_rank()
    except Exception:
        return int(os.getenv("LOCAL_RANK", "0"))

def _ws():
    try:
        return torch.distributed.get_world_size()
    except Exception:
        return int(os.getenv("WORLD_SIZE", "1"))

# ========== 0) 环境与关键信息 ==========
def print_runtime_env(args, tag="TRAIN"):
    r = _rank()
    print(
        f"[{tag}][R{r}] world_size(env)={os.getenv('WORLD_SIZE')}  "
        f"world_size(args)={getattr(args,'world_size', 'NA')}  "
        f"local_rank={getattr(args,'local_rank','NA')}  "
        f"device={getattr(args,'device','NA')}  "
        f"deepspeed={bool(getattr(args,'deepspeed', None))}  "
        f"grad_acc={getattr(args,'gradient_accumulation_steps','NA')}"
    )

# ========== 1) 训练 DataLoader/采样器体检（证明“有没有数据”） ==========
def inspect_train_dataloader(trainer):
    """
    在 train 前调用。只打印信息，不消耗数据。
    """
    r = _rank()
    try:
        dl = trainer.get_train_dataloader()
    except Exception as e:
        print(f"[DL][R{r}] get_train_dataloader() failed: {repr(e)}")
        return

    # 基本规模
    try:
        ds_len = len(getattr(dl, "dataset", []))
    except Exception:
        ds_len = "NA"
    try:
        dl_len = len(dl)
    except Exception:
        dl_len = "NA"

    bs = getattr(dl, "batch_size", None)
    print(f"[DL][R{r}] len(dataset)={ds_len}  len(dataloader)={dl_len}  batch_size={bs}")

    # 采样器信息
    sampler = getattr(dl, "sampler", None)
    if sampler is None:
        print(f"[DL][R{r}] sampler=None")
        return

    fields = {
        "num_replicas": getattr(sampler, "num_replicas", None),
        "rank": getattr(sampler, "rank", None),
        "num_samples": getattr(sampler, "num_samples", None),
        "total_size": getattr(sampler, "total_size", None),
        "drop_last": getattr(sampler, "drop_last", None),
    }
    brief = ", ".join([f"{k}={v}" for k, v in fields.items()])
    print(f"[DL][R{r}] sampler: {sampler.__class__.__name__} ({brief})")

    # 若有 indices（自定义/分桶采样器常见），打印前几个
    idxs = getattr(sampler, "indices", None)
    if idxs is not None:
        try:
            nidx = len(idxs)
            head = list(idxs[:16])
            print(f"[DL][R{r}] sampler.indices_len={nidx}, head={head}")
        except Exception:
            print(f"[DL][R{r}] sampler.indices=<?>")

    # 判定空分片（最关键）
    if isinstance(dl_len, int) and dl_len == 0:
        print(f"[DL][R{r}][ALERT] dataloader is EMPTY -> 该 rank 无 batch（疑似二次分片或切分参数错误）")
    if fields["num_samples"] == 0:
        print(f"[DL][R{r}][ALERT] sampler.num_samples=0 -> 该 rank 无样本（高概率二次分片）")

# ========== 2) 统计每个 rank 实际跑了多少步 ==========
class StepCounter(TrainerCallback):
    def __init__(self, local_rank):
        self.rank = local_rank
        self.cnt = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.cnt += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        # 每个 rank 各自打印，无需聚合
        print(f"[STEPS][R{self.rank}] steps_in_epoch={self.cnt}")

# ========== 3) 每步耗时（前若干步），判断是否 CPU/IO 卡住 ==========
class WallClockTimer(TrainerCallback):
    """
    仅打印前 n_steps 步的 wall time；若 R0 明显慢于其它 rank，多半被 CPU/IO 阻塞。
    """
    def __init__(self, local_rank, n_steps=5):
        self.rank = local_rank
        self.n_steps = n_steps
        self._t = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < self.n_steps:
            self._t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.n_steps and self._t is not None:
            dt = time.time() - self._t
            print(f"[TIMER][R{self.rank}] step {state.global_step} wall_time={dt:.3f}s")

# ========== 4) CUDA/CPU 活动热身采样（可选，极轻量） ==========
class TinyProfiler(TrainerCallback):
    """
    仅采样 very few steps 的 CPU/CUDA 活动总时长对比（不输出 TB trace，尽量轻）。
    若看到 R0 CUDA 时间 ~0 而其它 rank CUDA 时间很高 -> R0 在做 CPU/IO 工作。
    """
    def __init__(self, local_rank, active_steps=(1, 2, 3)):
        self.rank = local_rank
        self.active_steps = set(active_steps)
        self._cpu = 0.0
        self._cuda = 0.0
        self._t_cpu = None
        self._t_cuda = None
        self._cuda_sync = torch.cuda.synchronize

    def on_step_begin(self, args, state, control, **kwargs):
        s = state.global_step
        if s in self.active_steps:
            self._t_cpu = time.time()
            torch.cuda.synchronize()
            self._t_cuda = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        s = state.global_step
        if s in self.active_steps and self._t_cpu is not None:
            torch.cuda.synchronize()
            t_cuda_end = time.time()
            cpu_time = t_cuda_end - self._t_cpu
            cuda_time = t_cuda_end - self._t_cuda
            self._cpu += cpu_time
            self._cuda += cuda_time
            print(f"[TPR][R{self.rank}] step {s} cpu_time≈{cpu_time:.3f}s cuda_time≈{cuda_time:.3f}s")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"[TPR][R{self.rank}] SUM cpu≈{self._cpu:.3f}s  cuda≈{self._cuda:.3f}s (steps={sorted(self.active_steps)})")

# ========== 5) 一键挂载 ==========
def attach_debug_probes(trainer, training_args, n_timer_steps=5, enable_tiny_profiler=True):
    print_runtime_env(training_args, tag="TRAIN")
    # DataLoader 体检（训练开始前调用）
    inspect_train_dataloader(trainer)

    # 回调挂载
    trainer.add_callback(StepCounter(training_args.local_rank))
    trainer.add_callback(WallClockTimer(training_args.local_rank, n_steps=n_timer_steps))
    if enable_tiny_profiler:
        trainer.add_callback(TinyProfiler(training_args.local_rank, active_steps=(1,2,3)))
