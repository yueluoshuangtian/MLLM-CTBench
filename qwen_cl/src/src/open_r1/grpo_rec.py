# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
# ✅ Compat shim: flash-attn 2.8+ uses torch.library.wrap_triton (torch >= 2.6).
# On this torch version we polyfill it as a passthrough so flash_attn's triton
# kernels launch via plain kernel[grid](...). Must run before any flash_attn import.
import torch as _torch_for_shim
if not hasattr(_torch_for_shim.library, "wrap_triton"):
    def _wrap_triton_passthrough(kernel):
        return kernel
    _torch_for_shim.library.wrap_triton = _wrap_triton_passthrough
del _torch_for_shim

from rouge import Rouge
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
import sys
directory_path ='/home/houzhiyan/qwen/src/src/open_r1'
sys.path.insert(0, directory_path)
from trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

from reward import edit_distance_accuracy,levenshtein_distance,contains_special_symbols,find_words_num
# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
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

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False




def load_json_datas(json_path,sampling_strategy,sampling_number):
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

    if ":" in sampling_strategy:
        sampling_strategy, sampling_number = sampling_strategy.split(":")
        if "%" in sampling_number:
            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
        else:
            sampling_number = int(sampling_number)

    # Apply the sampling strategy
    if sampling_strategy == "first" and sampling_number is not None:
        cur_data_dict = cur_data_dict[:sampling_number]
    elif sampling_strategy == "end" and sampling_number is not None:
        cur_data_dict = cur_data_dict[-sampling_number:]
    elif sampling_strategy == "random" and sampling_number is not None:
        random.shuffle(cur_data_dict)
        cur_data_dict = cur_data_dict[:sampling_number]
    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
    return cur_data_dict

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    cur_data_dict = load_json_datas(json_path,sampling_strategy,sampling_number)
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".json"):
            json_path = data_path
            sampling_strategy = "all" if script_args.sampling_strategy is None else script_args.sampling_strategy
            sampling_number = None
            cur_data_dict = load_json_datas(json_path,sampling_strategy,sampling_number)
            self.list_data_dict.extend(cur_data_dict)

        else:
            raise ValueError(f"Unsupported file type: {data_path}")

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
            return {
                "prompt": [

                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"].split('prompt:\n')[0],Question_propmt = example["problem"].split('prompt:\n')[-1].lower().rstrip('.?'))}
                        ],
                    },
                ]
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
        

        return {
            'image': image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
        }



def format_reward(completions, **kwargs):
    
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>\s*<answer>([\s\S]*?)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, solution, **kwargs):
    
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    if isinstance(completions[0],str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
        format_pattern = r"<think>.*?</think>\s*<answer>([\s\S]*?)</answer>"
        match = re.fullmatch(format_pattern, content, re.DOTALL)
        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>([\s\S]*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>([\s\S]*?)</answer>', content)
                student_answer = content_match.group(1).strip().lower() if content_match else content.strip()
                
                if student_answer == ground_truth:
                    reward = 1.0
                # pdb.set_trace()
                
                #计算latex格式
                if contains_special_symbols(ground_truth):
                    accuracy = edit_distance_accuracy(student_answer, ground_truth)
                    reward = accuracy
                    
                #计算长输出
                elif len(find_words_num(ground_truth))>=2:
                    
                    rouger = Rouge()
                    rouge_scores = rouger.get_scores(extract_words(student_answer), extract_words(ground_truth), avg=True)
                    reward = max(rouge_scores['rouge-l']['f'],0.0)
                    # pdb.set_trace()
                if extract_letters(student_answer)[-1] == ground_truth:
                    reward = 1.0
                # Compare the extracted answers
                
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            # 优先使用 LOG_DIR，其次用当前工作目录
            log_dir = os.getenv("LOG_DIR", os.getcwd())
            os.makedirs(log_dir, exist_ok=True)
            local_rank = os.getenv("LOCAL_RANK", "0")  # torchrun 会注入
            log_path = os.path.join(log_dir, f"grpo_debug_rank{local_rank}.log")
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            except Exception:
                pass  # 绝不影响训练
    return rewards

def extract_letters(text): # for RAVEN
    pattern = r'(^|\s|\[|\()([A-H])(\s|\]|\)|$)'
    matches = re.findall(pattern, text)
    return [match[1] for match in matches]
def extract_words(text):
    """
    从文本中提取所有有意义的词（去除标点、符号、括号等）。
    """
    # 只保留英文单词，忽略符号、引号、标点、括号等
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return ' '.join(words)
reward_funcs_registry_ohthers = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}
from transformers import TrainerCallback, TrainerControl, TrainerState
from pathlib import Path

def main(script_args, training_args, model_args):

    reward_funcs = [reward_funcs_registry_ohthers[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
   
    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    
    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
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


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
