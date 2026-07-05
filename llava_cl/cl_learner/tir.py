import os
import torch
import torch.distributed as dist

from .ewc import EWCLearner
from .mas import MASLearner


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class TIRLearner:

    Id = None
    task_sim = None

    def __init__(self, task_encoder):
        self.task_encoder = task_encoder

    def update_Ipt(self):
        rank0_print("Updating Ipt.")
        self.normalize_Ipt()
        if self.Ipt is not None:
            self.Id = torch.load(self.save_file, map_location="cpu")["Id"]
            self.Id = {n: p.to(torch.int8) for n, p in self.Id.items()}
            for n, p in self.curIpt.items():
                p = p.cpu() / self.iters
                d = self.Ipt[n]
                self.Ipt[n] = torch.max(p, d)
                self.Id[n][p > d] = self.num_seen_tasks
        else:
            self.Ipt = {n: p.cpu() / self.iters for n, p in self.curIpt.items()}
            self.Id = {n: torch.zeros_like(p).char() for n, p in self.Ipt.items()}
        self.curIpt = None
        
    #这里是归一化，所有的值加起来为1
    def normalize_Ipt(self):
        totalIpt = 0.
        for p in self.curIpt.values():
            totalIpt += p.sum()
        self.curIpt = {n: p / totalIpt for n, p in self.curIpt.items()}

    def save(self):
        rank0_print("Saving Ipt and Id...")
        if local_rank <= 0:
            os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
            torch.save({"Ipt": self.Ipt, "Id": self.Id, "num_seen_tasks": self.num_seen_tasks+1}, self.save_file)
        dist.barrier()

    def load(self):
        if os.path.isfile(self.save_file) and self.Ipt is None:
            rank0_print(f"Loading Ipt and Id from {self.save_file}")
            ckpt = torch.load(self.save_file, map_location="cpu")
            self.Ipt = {n: p.to(torch.bfloat16) for n, p in ckpt["Ipt"].items()}
            self.Id = {n: p.to(torch.int8) for n, p in ckpt["Id"].items()}
            self.load_num_seen_tasks = ckpt["num_seen_tasks"]
            del ckpt

    def before_train(self, task_id, model, tokenizer=None, train_dataset=None, data_collator=None, **kwargs):
        self.load()
        if self.num_seen_tasks >= self.load_num_seen_tasks:
            task_sim = self.task_encoder.cal_task_sim(self.num_seen_tasks, train_dataset, data_collator)
            if task_id > 0 and self.Ipt is not None:
                task_sim = task_sim.cpu()
                for n, p in self.Ipt.items():
                    self.Ipt[n] = self.Ipt[n] * (1. - task_sim)[self.Id[n].int()]
                    # self.Ipt[n] = self.Ipt[n] * (1. - task_sim)[torch.randint(task_sim.shape[0], (1,))[0]]
                    # self.Ipt[n] = self.Ipt[n] * task_sim[self.Id[n].int()]
                self.Id = None  # save memory
                self.Ipt = {n: p.to(dtype=torch.bfloat16, device=self.device) for n, p in self.Ipt.items()}
                self.get_Whist(model)


class TIREWCLearner(TIRLearner, EWCLearner):
    def __init__(self, task_encoder, trainer_cls, lbd, rank, output_dir, training_args):
        EWCLearner.__init__(self, trainer_cls, lbd, rank, output_dir, training_args)
        TIRLearner.__init__(self, task_encoder)
        global local_rank
        local_rank = rank


class TIRMASLearner(TIRLearner, MASLearner):
    def __init__(self, task_encoder, trainer_cls, lbd, rank, output_dir, training_args):
        MASLearner.__init__(self, trainer_cls, lbd, rank, output_dir, training_args)
        TIRLearner.__init__(self, task_encoder)
        global local_rank
        local_rank = rank
