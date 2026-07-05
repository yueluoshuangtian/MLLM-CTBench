import os
import gc
import copy
import glob
import torch
import torch.distributed as dist
from .base import BaseCLearner

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def release_memory():
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats()


def memory_stats():
    rank0_print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
    rank0_print(f"memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2}")


class LogitsSaver(BaseCLearner):
    id_logits_dict = {}

    def loss(self, loss, logits=None, sample_ids=None, **kwargs):
        # smoke 阶段 trainer 不传 sample_ids, 跳过保存 (DER 蒸馏不会实际生效, 但 wiring 通).
        # §7.2 真要 DER 数值正确, 需 collator 把 sample_id 注入 inputs 并由 trainer 透传.
        if logits is None or sample_ids is None:
            return loss
        logits = logits.detach().cpu().to(dtype=torch.bfloat16)
        for i, sample_id in enumerate(sample_ids):
            self.id_logits_dict[sample_id] = logits[i]
        return loss
    

class DERLearner(BaseCLearner):
    num_seen_tasks = 0

    def __init__(self, trainer_cls, lbd, rank, output_dir, training_args):
        global local_rank
        local_rank = rank

        self.lbd = lbd
        self.device = torch.device("cuda", local_rank)
        self.save_file = os.path.join(output_dir, "{task_id}_cl_states_{rank}.bin")

        self.training_args = copy.deepcopy(training_args)
        self.training_args.learning_rate = 0
        self.train_cl_learner = LogitsSaver()

        self.trainer_cls = trainer_cls

    def loss(self, loss, logits=None, labels=None, replay_logits=None, **kwargs):
        if replay_logits is not None and logits is not None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # shift to mask the right tokens using "labels"
            logits = logits[..., :-1, :].contiguous()
            replay_logits = replay_logits[..., :-1, :].contiguous().float()
            labels = labels[..., 1:].contiguous()
            # Note that the replay_logits and logits may be padded to different lengths
            # But use mask to consider only the non-padded tokens can avoid such inconsistency
            token_indices = torch.nonzero(labels != -100, as_tuple=True)  # All right-padded so the label token indices are the same
            distill_loss = self.lbd * loss_fct(logits[token_indices], replay_logits[token_indices].softmax(-1))
        else:
            distill_loss = 0
        rank0_print(f"Supervise Loss: {loss} Distill Loss: {distill_loss}")
        return loss + distill_loss

    def before_train(self, task_id, model, tokenizer, train_dataset=None, replay_dataset=None, data_collator=None, **kwargs):
        if task_id > 1:
            if not os.path.exists(self.save_file.format(task_id=task_id-1, rank=local_rank)):
                assert replay_dataset is not None, f"DER task {task_id} requires replay_dataset (prior-task data)"
                release_memory()
                self.train_cl_learner.id_logits_dict = {}
                trainer = self.trainer_cls(cl_learner=self.train_cl_learner,
                                        train_dataset=replay_dataset,
                                        data_collator=data_collator,
                                        args=self.training_args,
                                        model=model,
                                        tokenizer=tokenizer)
                trainer.train()
                self.save(task_id)
                del trainer
                release_memory()
            
            id_logits_dict = self.load(task_id)   # load logits for previous replay datasets
            train_dataset.id_logits_dict = id_logits_dict        

    def save(self, task_id):
        rank0_print("Saving replay logits.")
        if local_rank <= 0:
            os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        torch.save(self.train_cl_learner.id_logits_dict, self.save_file.format(task_id=task_id-1, rank=local_rank))
        dist.barrier()

    def load(self, task_id):
        id_logits_dict = {}
        for tid in range(task_id):
            for save_file in glob.glob(self.save_file.format(task_id=tid, rank="*")):
                rank0_print(f"Loading replay logits from {save_file} ...")
                id_logits_dict.update(torch.load(save_file, map_location="cpu"))
        return id_logits_dict
    