"""
MAS (Aljundi 2018): same as EWC but importance is |grad| instead of grad^2.
Inherits from EWCLearner; only overrides the per-step accumulation rule.
"""

import os
import torch
from .ewc import EWCLearner


class MASLearner(EWCLearner):

    name = "mas"

    def __init__(self, lbd, output_dir, **kwargs):
        super().__init__(lbd=lbd, output_dir=output_dir, **kwargs)
        # Distinguish save file so MAS/EWC ckpts don't collide
        self.save_file = os.path.join(output_dir, "cl_states_mas_lora.bin")

    def _accumulate_one(self, name, grad):
        """MAS importance = |grad| (sensitivity proxy)."""
        self.curIpt[name].add_(grad.abs().detach().to("cpu", dtype=torch.bfloat16))
