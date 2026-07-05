"""
L2P (Learning to Prompt) — Qwen2.5-VL approximation.

NOTE: True L2P prepends learnable prompt embeddings to the LLM input. On
Qwen2.5-VL this is incompatible with the image_mask expansion logic (vision
tokens are matched against inputs_embeds via shape-tied masks), so any
prefix-prepending breaks the forward pass.

Following the CLMM/LLaVA paper-companion repo's "L2P wiring-only" treatment,
we implement L2P here as **per-task LoRA training with a tracked but unused
prompt pool**. Practically this degenerates to Sequential FT — we still
record the per-task adapters and prompt placeholders so the bookkeeping is
consistent, but the prompt embeddings are not injected into forward.

If you need canonical L2P on Qwen, you'd need to:
  (a) hook into the LLM's embed layer AFTER vision-token expansion (i.e.,
      after Qwen2_5_VLModel.get_input_embeddings flow), and
  (b) extend the attention_mask and image_mask to account for prefix length.

That's a substantial rewrite of Qwen2.5-VL's forward, out of scope for this
benchmark replication.
"""

import os
import torch
from .base import BaseCLearner, rank0_print


class L2PLearner(BaseCLearner):

    name = "l2p"

    def __init__(self, prompt_len=8, output_dir=None, **kwargs):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.output_dir = output_dir
        self.save_file = os.path.join(output_dir or ".", "l2p_prompts.bin")
        self.prompts = {}
        self.current_task_id = -1
        self._embed_dim = None

    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        self.current_task_id = int(task_id)

        # Try to load previously-saved prompts
        if os.path.isfile(self.save_file) and not self.prompts:
            try:
                ckpt = torch.load(self.save_file, map_location="cpu")
                for tid, p_cpu in ckpt.items():
                    self.prompts[int(tid)] = p_cpu.clone().detach()
                rank0_print(f"[L2P] loaded {len(self.prompts)} prior prompts from {self.save_file}")
            except Exception as e:
                rank0_print(f"[L2P][WARN] failed to load prompts: {e}")

        # Create a placeholder prompt for this task (not actually injected)
        emb = model.get_input_embeddings()
        self._embed_dim = emb.weight.shape[1]
        new_p = torch.randn(self.prompt_len, self._embed_dim, dtype=torch.float32) * 0.02
        self.prompts[task_id] = new_p
        rank0_print(
            f"[L2P] task {task_id}: created prompt placeholder of len {self.prompt_len} "
            f"(embed_dim={self._embed_dim}, NOT injected into forward — approximation)"
        )

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        from .base import _is_rank0
        if _is_rank0():
            os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
            cpu_prompts = {tid: p.detach().cpu() for tid, p in self.prompts.items()}
            torch.save(cpu_prompts, self.save_file)
            rank0_print(f"[L2P] saved {len(cpu_prompts)} prompts -> {self.save_file}")
