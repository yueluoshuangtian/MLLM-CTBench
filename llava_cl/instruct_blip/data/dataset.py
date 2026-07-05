import os
import math
import copy
import json
from PIL import Image
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import transformers

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


@dataclass
class DataArguments:
    data_dir: Optional[str] = field(default="./playground/data/blip_train_data/",
                                    metadata={"help": "Path to the json directory"})
    tasks: Optional[str] = field(default=None,
                                 metadata={"help": "Path to the training data."})
    initial_tasks: Optional[str] = field(default="",
                                         metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    replay_ratio: float = 0.01


class TestDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image = Image.open(os.path.join(self.image_folder, line["image"])).convert('RGB')
        data_dict = {"image": image,
                     "prompt": line["text"],
                     "question_id": line["question_id"]}

        return data_dict

    def __len__(self):
        return len(self.questions)


@dataclass
class DataCollatorForTestDataset(object):
    def __init__(self, processor: transformers.InstructBlipProcessor):
        self.processor = processor

    def __call__(self, instances: Sequence[Dict]) -> transformers.image_processing_utils.BatchFeature:
        images, prompts, question_ids = tuple([instance[key] for instance in instances] for key in ("image", "prompt", "question_id"))
        batch = self.processor(images=images,
                               text=prompts,
                               return_tensors="pt")
        batch["question_ids"] = question_ids
        batch["prompts"] = prompts
        return batch


def create_test_loader(questions, image_folder, processor, batch_size=1, num_workers=4):
    dataset = TestDataset(questions, image_folder)
    data_collator = DataCollatorForTestDataset(processor=processor)
    data_loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 replay_data_paths: list = None):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        if replay_data_paths is not None:
            for path in replay_data_paths:
                list_data = json.load(open(path, "r"))
                len_data = len(list_data)
                len_selected = int(len_data * data_args.replay_ratio)
                selected = np.random.choice(np.arange(len_data, dtype=np.int64), len_selected, replace=False)
                list_data_dict.extend(list_data[i] for i in selected)
                rank0_print(f"Loaded old dataset from {path}")

        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            length_list.append(len(sample["prompt"].split()) + len(sample["answer"].split()))
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        image = Image.open(os.path.join(self.data_args.image_folder, sources["image"])).convert('RGB')
        data_dict = {"image": image,
                     "prompt": sources["prompt"],
                     "answer": sources["answer"], }
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):

    def __init__(self, processor: transformers.InstructBlipProcessor, num_query_tokens: int):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.processor.tokenizer.truncation_side = "right"
        self.num_query_tokens = num_query_tokens

    def __call__(self, instances: Sequence[Dict]) -> transformers.image_processing_utils.BatchFeature:
        images, prompts, answers = tuple(
            [instance[key] for instance in instances] for key in ("image", "prompt", "answer"))
        convs = [prompt + " " + answer + " " + self.processor.tokenizer.eos_token for prompt, answer in
                 zip(prompts, answers)]
        batch = self.processor(images=images,
                               text=convs,
                               padding=True,
                               truncation=True,
                               return_tensors="pt")
        prompts_batch = self.processor(text=prompts,
                                       padding=True,
                                       return_tensors="pt")
        batch["qformer_input_ids"] = prompts_batch["qformer_input_ids"]
        batch["qformer_attention_mask"] = prompts_batch["qformer_attention_mask"]

        labels = batch["input_ids"].masked_fill(batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100)

        # do not apply loss to the text inputs
        for i, l in enumerate(prompts_batch["attention_mask"].sum(1)):
            labels[i, :l] = -100

        # do not apply loss to the query tokens
        empty_labels = torch.ones((labels.size(0), self.num_query_tokens), dtype=torch.long).fill_(-100)
        labels = torch.cat([empty_labels, labels], dim=1)

        batch["labels"] = labels
        batch["prompts"] = prompts
        batch["answers"] = answers
        return batch


def make_supervised_data_module(processor: transformers.InstructBlipProcessor,
                                data_args: DataArguments,
                                num_query_tokens: int,
                                rank: int) -> Dict:
    global local_rank
    local_rank = rank

    assert isinstance(data_args.tasks, str), "data_args.tasks must be a string"
    data_path = os.path.join(data_args.data_dir, f"{data_args.tasks}.json")
    train_dataset = LazySupervisedDataset(data_path=data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(processor=processor, num_query_tokens=num_query_tokens)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_cl_data_module(processor: transformers.InstructBlipProcessor,
                        data_args: DataArguments,
                        task_list: str,
                        num_query_tokens: int,
                        rank: int) -> List[Dict]:
    global local_rank
    local_rank = rank

    tasks = [task.strip() for task in task_list.split(",") if len(task.strip()) > 0]
    data_paths = [os.path.join(data_args.data_dir, f"{task}.json") for task in tasks]

    if data_args.initial_tasks is not None:
        initial_tasks = [task.strip() for task in data_args.initial_tasks.split(',') if len(task.strip()) > 0]
        initial_data_paths = [os.path.join(data_args.data_dir, f"{task}.json") for task in initial_tasks]
    else:
        initial_data_paths = []

    data_collator = DataCollatorForSupervisedDataset(processor=processor, num_query_tokens=num_query_tokens)

    data_modules = []
    rank0_print(f"Replay ratio: {data_args.replay_ratio}")
    for task_id, cur_data_path in enumerate(data_paths):
        rank0_print(f"Loading dataset from {cur_data_path}")
        train_dataset = LazySupervisedDataset(data_path=cur_data_path,
                                              data_args=data_args,
                                              replay_data_paths=initial_data_paths + data_paths[:task_id]
                                              if data_args.replay_ratio > 0 else None)
        data_modules.append(
            dict(train_dataset=train_dataset,
                 eval_dataset=None,
                 data_collator=data_collator, )
        )
    return data_modules
