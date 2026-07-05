"""
CL learner factory: build_cl_learner(method, ...) -> BaseCLearner.

Methods supported (paper §IV-A):
  - none        : pure sequential SFT (no extra reg, no replay)
  - ewc         : Elastic Weight Consolidation
  - mas         : Memory Aware Synapses (EWC with |grad| importance)
  - lwf         : Learning without Forgetting (teacher KL)
  - freeze_init : freeze early LLM blocks (LoRA on later blocks only)
  - freeze_last : freeze later LLM blocks (LoRA on earlier blocks only)
  - replay      : Experience Replay (mix prior data via dataset.py)
  - der         : Dark Experience Replay (MSE on prior-model logits)
  - l2p         : Learning to Prompt (simplified, per-task prefix)
  - max_merge   : MagMaX — independent FT per task + element-wise max merge

Lazy import per branch so a broken import in one method (e.g. moelora's CoIN
fork) doesn't take down the others.
"""

from .base import BaseCLearner


def build_cl_learner(method, *, output_dir, lbd=None, replay_paths=None,
                     replay_ratio=0.0, prompt_len=8, **kwargs):
    method = (method or "none").lower()

    if method in ("none", "sft", "sequential", ""):
        return BaseCLearner()

    if method == "ewc":
        from .ewc import EWCLearner
        return EWCLearner(lbd=lbd if lbd is not None else 1e3, output_dir=output_dir)

    if method == "mas":
        from .mas import MASLearner
        return MASLearner(lbd=lbd if lbd is not None else 1e2, output_dir=output_dir)

    if method == "lwf":
        from .lwf import LwFLearner
        return LwFLearner(lbd=lbd if lbd is not None else 1.0)

    if method == "freeze_init":
        from .freeze import FreezeLearner
        return FreezeLearner(variant="init")

    if method == "freeze_last":
        from .freeze import FreezeLearner
        return FreezeLearner(variant="last")

    if method == "freeze":  # alias defaults to freeze_last (paper's better variant)
        from .freeze import FreezeLearner
        return FreezeLearner(variant="last")

    if method == "replay":
        from .replay import ReplayLearner
        return ReplayLearner(replay_paths=replay_paths, replay_ratio=replay_ratio)

    if method == "der":
        from .der import DERLearner
        return DERLearner(lbd=lbd if lbd is not None else 0.5)

    if method == "l2p":
        from .l2p import L2PLearner
        return L2PLearner(prompt_len=prompt_len, output_dir=output_dir)

    if method == "max_merge":
        from .max_merge import MaxMergeLearner
        return MaxMergeLearner(output_dir=output_dir)

    if method == "moelora":
        from .moelora import moeloraLearner
        return moeloraLearner(**kwargs)

    raise ValueError(f"unknown cl_method: {method}")
