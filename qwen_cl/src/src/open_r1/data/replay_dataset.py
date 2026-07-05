import yaml
import json
import math
import random
import os
from typing import Dict, Optional, Sequence, List
import logging
from trl import  ScriptArguments
from dataclasses import dataclass, field

from torch.utils.data import Dataset
logger = logging.getLogger(__name__) 

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default='/home/houzhiyan/dataset/images', metadata={"help": "The root directory of the image."})
    sampling_strategy: str = field(default="all", metadata={"help": "加载数据时采样的策略."})
    is_vqa: bool = field(default=True, metadata={"help": "训练数据的类型."})
    #用于训练的数据的位置
    # data_dir: Optional[str] = field(default="/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test",
    #                                 metadata={"help": "Path to the json directory"})
    tasks: Optional[str] = field(default=None,
                                 metadata={"help": "Path to the training data."})
    replay_ratio: Optional[float]= field(default=0.0,
                                 metadata={"help": "持续学习方法的参数"})
    cl_method:Optional[str] = field(default=None,
                                 metadata={"help": "持续学习方法"})
    cl_method_alpha:Optional[float] = field(default=0.0,
                                 metadata={"help": "持续学习方法的参数"})
    non_duplicates:Optional[bool] = field(default=True)



def load_json_datas(json_path,sampling_strategy,sampling_ratio):
    if json_path.endswith(".jsonl"):
        cur_data_dict = []
        with open(json_path, "r") as json_file:
            for line in json_file:
                cur_data_dict.append(json.loads(line.strip()))
    elif json_path.endswith(".json"):
        with open(json_path, "r") as json_file:
            cur_data_dict = json.load(json_file)
    else:
        raise ValueError(f"Unsupported file type: {json_path}")

    # if ":" in sampling_strategy:
    #     sampling_strategy, sampling_number = sampling_strategy.split(":")
    #     if "%" in sampling_number:
    #         sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
    #     else:
    #         sampling_number = int(sampling_number)

    # Apply the sampling strategy
    total_num = len(cur_data_dict)
    if sampling_strategy == 'all':
        return cur_data_dict
    if sampling_strategy == "first" and sampling_ratio is not None:
        cur_data_dict = cur_data_dict[:int(total_num *sampling_ratio)]
    elif sampling_strategy == "end" and sampling_ratio is not None:
        cur_data_dict = cur_data_dict[-int(total_num *sampling_ratio):]
    elif sampling_strategy == "random" and sampling_ratio is not None:
        random.shuffle(cur_data_dict)
        cur_data_dict = cur_data_dict[:int(total_num *sampling_ratio)]
    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
    return cur_data_dict
def freeze_model(model,layers):
    for name, param in model.named_parameters():
        if any(name.startswith(f"model.layers.{i}") for i in layers):
            param.requires_grad = False
class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        script_args: SFTScriptArguments,
        replay_data_paths: list | None = None,
        fixed_datas: dict | None = None, 
    ):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.list_data_dict.extend(load_json_datas(data_path, 'all' if script_args.sampling_strategy is None else script_args.sampling_strategy, None))
                
        if replay_data_paths is not None and script_args.replay_ratio > 0:

            if fixed_datas is not None:                     # 固定回放
                for p in replay_data_paths:
                    self.list_data_dict.extend(fixed_datas[p])
            else:
                sampling_strategy = 'random'
                sampling_ratio = script_args.replay_ratio
                for cur_data_path in replay_data_paths:
                    
                    self.list_data_dict.extend(load_json_datas(cur_data_path,sampling_strategy,sampling_ratio))

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i):
        # 处理批量索引（列表）
        if isinstance(i, list):
            return [self._get_single_item(idx) for idx in i]
        # 处理单个索引（整数）
        return self._get_single_item(i)
    
    def _get_single_item(self, i):

        # Format into conversation
        def make_conversation(example):
            QUESTION_TEMPLATE = "{Question}"
            return [{
                "role": "user",
                "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])}]
                },
                {
                    
                        "role": "assistant",
                        "content": example['solution'],
                       
                }
            ]
        
        # Format into conversation
        def make_conversation_image(example):
            QUESTION_TEMPLATE = "{Question}"

            image_root = self.script_args.image_root
            # print(111, image_root)
            # print(222, example['image'])
            image_path = os.path.join(image_root, example['image'])

            return  [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": example['solution'],
                    }
                ]

        example = self.list_data_dict[i]
        if "image" in example.keys():
            example["messages"] = make_conversation_image(example)
        else:
            example['messages'] = make_conversation(example)
        return example

def make_cl_data_module(script_args:SFTScriptArguments) -> List[Dict]:
    #data_args.dataset_name是包含所有训练集的文件夹路径，tasks是sh文件中输入的
    #data_paths包含任务json文件的位置的list
    data_paths = [os.path.join(script_args.dataset_name,f'{task}.json') for task in script_args.tasks]

    data_modules = []
    for task_id,cur_data_path in enumerate(data_paths):
    
        replay_paths = data_paths[:task_id] if script_args.cl_method in ['replay', 'der'] else None
        train_dataset = LazySupervisedDataset(cur_data_path, 
                                              script_args,
                                              replay_paths)

        data_modules.append(dict(train_dataset=train_dataset,
                                 task_name = script_args.tasks[task_id]))
        logger.info(f"成功加载{task_id}任务的数据集")
    
    return data_modules
