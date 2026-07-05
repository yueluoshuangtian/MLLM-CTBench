import copy
import torch
from .base import BaseCLearner


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class LwFLearner(BaseCLearner):

    def __init__(self, lbd, rank, model):
        global local_rank
        local_rank = rank

        self.lbd = lbd
        self.device = torch.device("cuda", rank)
        self.model = copy.deepcopy(model)
        self.model.to(device=self.device, dtype=torch.bfloat16)

    def update_model(self, latest_model):
        for name, param in self.model.named_parameters():
            param.data.copy_(latest_model.state_dict()[name])

    def loss(self, loss, inputs=None, logits=None, labels=None, **kwargs):
        kldiv_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

        with torch.no_grad():
            teacher_logits = self.model(**inputs).logits

        # # shift to mask the right tokens using "labels"
        # logits = logits[..., :-1, :].contiguous()
        # teacher_logits = teacher_logits[..., :-1, :].contiguous()
        # labels = labels[..., 1:].contiguous()

        token_indices = torch.nonzero(labels != -100, as_tuple=True)  # pad right so the label token indices are the same

        log_p = torch.log_softmax(logits[token_indices], dim=1)
        q = torch.softmax(teacher_logits[token_indices], dim=1)

        distill_loss = self.lbd * kldiv_fct(log_p, q)
        rank0_print(f"Supervise Loss: {loss} Distill Loss: {distill_loss}")

        return loss + distill_loss

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        if task_id > 0:
            self.update_model(model)
