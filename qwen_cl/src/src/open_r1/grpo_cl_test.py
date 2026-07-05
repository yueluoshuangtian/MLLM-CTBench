# ────────────────────────────
# 标准库
# ────────────────────────────
import glob
import os
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pdb  # 若只在调试阶段使用，可在正式运行前删掉

# ────────────────────────────
# 第三方库
# ────────────────────────────
from PIL import Image
from torch.utils.data import Dataset
from accelerate.state import AcceleratorState
from trl import ModelConfig, TrlParser, get_peft_config, ScriptArguments
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2
import torch.distributed as dist
# ────────────────────────────
# 本地/项目内模块
# ────────────────────────────
from sft_CL import release_memory
from grpo_rec import custom_forward, format_reward, accuracy_reward
from trainer import Qwen2VLGRPOTrainer, GRPOConfig
from sft_CL import release_memory
from data.dataset import load_json_datas

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward
@dataclass
class GRPOScriptArguments_cl(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    sampling_strategy:Optional[str] = field(
        default=None,
        metadata={"help": "single json sampling strategy"},
    )
    tasks: Optional[str] = field(default=None,
                                 metadata={"help": "Path to the training data."})
    replay_ratio: Optional[float]= field(default=0.0,
                                 metadata={"help": "持续学习方法的参数"})
    cl_method:Optional[str] = field(default=None,
                                 metadata={"help": "持续学习方法"})
    cl_method_alptha:Optional[float] = field(default=0.0,
                                 metadata={"help": "持续学习方法的参数"})
    
class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments_cl,replay_data_paths:list = None):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if replay_data_paths is not None:
            cur_data_dict = load_json_datas(data_path,'all',None)
            self.list_data_dict.extend(cur_data_dict)
            sampling_strategy = 'random'
            sampling_ratio = script_args.replay_ratio
            for cur_data_path in replay_data_paths:
                self.list_data_dict.extend(load_json_datas(cur_data_path,sampling_strategy,sampling_ratio))

        else:
            json_path = data_path
            sampling_strategy = "all" if script_args.sampling_strategy is None else script_args.sampling_strategy
            sampling_number = None
            cur_data_dict = load_json_datas(json_path,sampling_strategy,sampling_number)
            self.list_data_dict.extend(cur_data_dict)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then {Question_propmt} in <answer> </answer> tags. Output the final answer in JSON format."
        
        def make_conversation(example):
            return {
            "prompt":
            [{
                "role": "user",
                "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"].split('prompt:\n')[0],Question_propmt = example["problem"].split('prompt:\n')[-1].lower().rstrip('.?'))}]
                },
                {
                    
                        "role": "assistant",
                        "content": example['solution'],
                       
                }]
            }
        # FIXME
        # This is only for Grounding task

        def make_conversation_image(example):
            # pdb.set_trace()
            return {
            "prompt":
            [{
                "role": "user",
                "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"].split('prompt:\n')[0],Question_propmt = example["problem"].split('prompt:\n')[-1].lower().rstrip('.?'))}]
                },
                {
                    
                        "role": "assistant",
                        "content": example['solution'],
                       
                }]
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
        else:
            image = None
        
        # pdb.set_trace()
        return {
            'image': image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt']
        }
import pdb
def make_cl_data_module(script_args:GRPOScriptArguments_cl) -> List[Dict]:
    # data_args.dataset_name是包含所有训练集的文件夹路径，tasks是sh文件中输入的
    # data_paths包含任务json文件的位置的list
    data_paths = [os.path.join(script_args.dataset_name,f'{task}.json') for task in script_args.tasks]

    data_modules = []
    for task_id,cur_data_path in enumerate(data_paths):
        
        train_dataset = LazySupervisedDataset(cur_data_path, 
                                              script_args,
                                              data_paths[:task_id] if script_args.cl_method in ['replay', 'der'] else None)

        data_modules.append(dict(train_dataset=train_dataset,
                                 task_name = script_args.tasks[task_id]))
        print(f"成功加载{task_id}任务的数据集")
    
    return data_modules
@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

reward_funcs_registry_ohthers = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

        
def cl_main(script_args, training_args, model_args):
    adapter_path = None
    script_args.tasks = [task.strip() for task in script_args.tasks.split(',') if len(task.strip())>0]

    #####decide start task:
    start_task_id = 0
    output_dir = training_args.output_dir
    os.makedirs(output_dir,exist_ok=True)
    for d in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isdir(d) and os.path.basename(d).isdigit() and len(os.listdir(d)) > 0:
            start_task_id = max(int(os.path.basename(d)), start_task_id)
    ##有文件夹有两种情况，训练完和没训练完
    maybe_exist_lora_model = os.path.join(output_dir,str(start_task_id))
    maybe_exist_lora_model_adapter = os.path.join(maybe_exist_lora_model,'adapter_config.json')
    if os.path.exists(maybe_exist_lora_model_adapter):
        #完成之后从下一个任务开始
        start_task_id += 1
    ##上述只要开始训练，start_task_id必然是>=1,所以两种情况，如果start_task_id=0就开始训练，如果start_task_id
    
    #之后加载模型时要记得改变model_args.model_name_or_path#
    
    data_modules = make_cl_data_module(script_args)
    reward_funcs = [reward_funcs_registry_ohthers[func] for func in script_args.reward_funcs]
    
    
    for task_id, data_module in enumerate(data_modules):
        trainer_cls = Qwen2VLGRPOTrainer
        if task_id + 1<start_task_id:
            continue
        
        if task_id > 0:
            adapter_path = os.path.join(output_dir,str(task_id))
        
        training_args.output_dir = os.path.join(output_dir,str(task_id + 1))  
        # Initialize the GRPO trainer,不同有三方面，初始模型不同，训练数据不同，保存位置不同
        # Initialize the GRPO trainer
        print(f'保存的位置为{training_args.output_dir},加载的adapter为{adapter_path}')
        pdb.set_trace()
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=data_module['train_dataset'],
            eval_dataset=None,
            peft_config=get_peft_config(model_args),
            adapter_path = adapter_path,
            freeze_vision_modules=model_args.freeze_vision_modules,
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
            torch_dtype=model_args.torch_dtype,
        )
        # Train and push the model to the Hub
        trainer.train()

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)
        del trainer
        release_memory()
        if dist.is_initialized():
            dist.destroy_process_group()

        AcceleratorState._reset_state(reset_partial_state=True)
                

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments_cl, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    cl_main(script_args, training_args, model_args)