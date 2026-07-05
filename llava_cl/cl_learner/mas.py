import torch
from .base import BaseCLearner
from .ewc import EWCLearner


class MASIptLearner(BaseCLearner):
    def loss(self, loss, logits=None, **kwargs):
        return logits.pow(2).sum(-1).mean()


class MASLearner(EWCLearner):

    def __init__(self, trainer_cls, lbd, rank, output_dir, training_args):
        super().__init__(trainer_cls, lbd, rank, output_dir, training_args)
        self.train_cl_learner = MASIptLearner()

    def cal_ipt(self, name):
        def hook(grad):
            grad = torch.nan_to_num(grad, nan=0)
            # curIpt 在 CPU (避免 7B full-FT 时 14GB curIpt 撑爆 backward).
            self.curIpt[name].add_(grad.abs().detach().to("cpu", dtype=torch.bfloat16))
        return hook
