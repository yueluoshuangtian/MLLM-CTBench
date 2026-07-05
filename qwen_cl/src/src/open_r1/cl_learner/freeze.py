"""
Freeze CL: just freeze a subset of LoRA layers per task.

Two variants follow the paper (MLLM-CTBench §IV):
- `freeze_init`: keep LoRA only on the LAST half of LLM transformer blocks
  (freeze "init" / lower blocks)
- `freeze_last`: keep LoRA only on the FIRST half of LLM transformer blocks
  (freeze "last" / output-side blocks)

Since LoRA is the only trainable surface, "freeze" here means zeroing
requires_grad on adapters in the unwanted block range.

This learner doesn't add reg loss; it just modifies which LoRA params
participate. before_train decides which to freeze. after_train is a no-op.
"""

import re
import torch
from .base import BaseCLearner, rank0_print


_LLM_BLOCK_RE = re.compile(r"\.layers\.(\d+)\.")


def _llm_block_idx(name: str):
    m = _LLM_BLOCK_RE.search(name)
    if m is None:
        return None
    return int(m.group(1))


class FreezeLearner(BaseCLearner):

    name = "freeze"

    def __init__(self, variant="last", **kwargs):
        """
        variant: 'init' freezes early blocks, 'last' freezes late blocks
        """
        super().__init__()
        assert variant in ("init", "last"), f"unknown freeze variant: {variant}"
        self.variant = variant

    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        # Find max LLM block idx among trainable LoRA params
        max_idx = -1
        block_ids = set()
        for n, p in model.named_parameters():
            if not (("lora_A" in n) or ("lora_B" in n)):
                continue
            bi = _llm_block_idx(n)
            if bi is None:
                continue
            block_ids.add(bi)
            if bi > max_idx:
                max_idx = bi
        if max_idx < 0:
            rank0_print("[Freeze][WARN] no LLM-block LoRA params found; freeze is no-op")
            return
        num_blocks = max_idx + 1
        half = num_blocks // 2

        if self.variant == "init":
            # freeze blocks [0, half) ; train [half, num_blocks)
            train_keep = lambda bi: bi is not None and bi >= half
            kept = f"[{half},{num_blocks})"
        else:  # "last"
            # freeze blocks [half, num_blocks) ; train [0, half)
            train_keep = lambda bi: bi is not None and bi < half
            kept = f"[0,{half})"

        n_off = 0
        n_keep = 0
        for n, p in model.named_parameters():
            if not (("lora_A" in n) or ("lora_B" in n)):
                continue
            bi = _llm_block_idx(n)
            if train_keep(bi):
                # leave it trainable
                n_keep += 1
            else:
                p.requires_grad_(False)
                n_off += 1
        rank0_print(
            f"[Freeze:{self.variant}] kept LoRA in LLM blocks {kept}: "
            f"trainable={n_keep}, frozen={n_off}, total_blocks={num_blocks}"
        )
