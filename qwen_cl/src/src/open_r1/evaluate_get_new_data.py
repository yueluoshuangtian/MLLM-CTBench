from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
import argparse
from torch.utils.data import DataLoader, Dataset
import pdb
# from caculate_accuracy import eval_all_tasks
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import glob
# import sys
# project_root = "/public/home/houzhiyan/VLM-R1-main"
# sys.path.append(project_root)
from eval.eval import eval_seqft_tasks
import time
data_model_types={
    "numglue":False,
    "art":True,
    "math":True,
    "fomc":False,
    "medical":True,
    "OCR":True,
    "science":True
}
question_file_dict = {
    "numglue": "/home/houzhiyan/dataset/qwen/infer/numglue.json",
    "art": "/home/houzhiyan/dataset/qwen/infer/art.json",
    "math": "/home/houzhiyan/dataset/qwen/infer/math.json",
    "fomc": "/home/houzhiyan/dataset/qwen/infer/fomc.json",
    "medical": "/home/houzhiyan/dataset/qwen/infer/medical.json",
    "OCR": "/home/houzhiyan/dataset/qwen/infer/OCR.json",
    "science": "/home/houzhiyan/dataset/qwen/infer/science.json"
}

#question_iamge
# image_folder = "/home/houzhiyan/dataset/images"
def prepare_chat_messages(data,task_val,image_root):
    """
    准备对话模型所需的输入消息列表
    
    参数:
    - DATA_ROOT: 数据集根目录
    - ds: 数据集名称（对应JSON文件名）
    - sample_num: 采样数量
    - IMAGE_ROOT: 图片存储根目录
    - SYSTEM_PROMPT: 可选系统提示
    
    返回:
    - 符合对话API格式的消息列表，包含图文混合输入和答案对
    
    功能:
    1. 加载并打乱指定数据集
    2. 构造多模态消息结构：
    - 含图片样本：用户消息包含图片路径+问题
    - 纯文本样本：构造用户问题+助理答案对
    3. 统一要求模型按照<think>推理+JSON格式回答
    """
    print(f"Processing {task_val}...")
    random.shuffle(data)
    #下面需要修改QUESTION_TEMPLATE和message
    QUESTION_TEMPLATE = """Original Question: {Question}.
        Original Answer: {Answer}.

        First, analyze the relationship between the question and its answer in <think> </think> tags. Explain how to rephrase the question while:
        1. Maintaining the same semantic meaning and expected answer
        2. Using different syntactic structures, vocabulary, and phrasing
        3. Preserving any key constraints or requirements

        Then, in <answer> </answer> tags, generate {num_variations} rephrased versions that would still lead to the same answer.

        Finally, return the result in JSON format with the key "rephrased_questions" containing a list of rephrased question strings.

        Example output:
        <think>
        The original question asks about [explain question intent] and requires [specific elements]. The answer confirms [key information]. Suitable rephrasing could include:
        - Changing sentence structure from [X] to [Y]
        - Replacing key terms like [A] with synonyms [B] or [C]
        - Adjusting the question format while maintaining [critical constraints]
        </think>
        <answer>
        {{
        "rephrased_questions": [
            "Rephrased version 1 that maintains the same meaning",
            "Alternative phrasing 2 that preserves the answer",
            "Different question structure 3 with identical semantic intent"
        ]
        }}
        </answer>"""


    messages = []

    for x in data:
        
        if "image" in x.keys():
            image_path = os.path.join(image_root, x['image'])
            message = [
                # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=x["problem"].rsplit("\n", 1)[0],
                                                         Answer=x["solution"],
                                                         num_variations = 4)
                    }
                ]
            }]
        else:
            message = [{
            "role": "user",
            "content": [{"type": "text", 
                         "text": QUESTION_TEMPLATE.format(Question=x["problem"].rsplit("\n", 1)[0],
                                                         Answer=x["solution"],
                                                         num_variations = 4)}]
            },
            {
                
                    "role": "assistant",
                    "content": x['solution'],
                    
            }
        ]
        messages.append(message)
    return messages

import gc
def release_memory():
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats()
def memory_stats():
    print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
    print(f"memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2}")
def evaluate_main(args,processor,model, task_val, output_path):
    

    #task_val是测试的任务名称
    task_val_filepath = question_file_dict[task_val]
    data = json.load(open(task_val_filepath, "r"))
    data = data[:args.sample_num]
    if not data:
        raise RuntimeError(f"[Warning] No data found for {task_val}")
        
    messages = prepare_chat_messages(data,
                                     task_val=task_val,
                                     image_root = args.image_root)
    all_outputs = []  # List to store all answers


    for i in tqdm(range(0, len(messages), args.bs)):
        batch_messages = messages[i:i + args.bs]
        
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        if data_model_types[task_val]:
            image_inputs, video_inputs = process_vision_info(batch_messages)
        else:
            image_inputs = None
            video_inputs = None
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")

        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=args.max_new_tokens, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        all_outputs.extend(batch_output_text)

    final_output = []
    for input_example, model_output in zip(data, all_outputs):
        original_output = model_output
        ground_truth = input_example['solution']
        if 'image' in input_example.keys():
            image_path = input_example["image"]
        else:
            image_path = None
        result = {
            'question': input_example['problem'],
            'ground_truth': ground_truth,
            'model_output': original_output,
            "image" : image_path,
            'question_id': input_example['question_id']
        }
        final_output.append(result)


    with open(output_path, "w") as f:
        json.dump({
            'results': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', required=True, type=str) # Base model: "Qwen/Qwen2-VL-2B-Instruct"
    parser.add_argument('--bs', default=32, type=int) # Batch size: reduce it if GPU OOM
    parser.add_argument("--image_root",default="/home/houzhiyan/dataset/images" ,type=str)
    parser.add_argument("--tasks", type=str)
    parser.add_argument("--sample_num", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--save_path",default="/home/houzhiyan/dataset/qwen" ,type=str)
    
    return parser.parse_args()
import pdb
if __name__ == "__main__":
    args = parse_arguments()
    
    tasks = [cur_task.strip() for cur_task in args.tasks.split(',') if len(cur_task.strip())>0]
    model_nums = 7
    ##代表是否在最后的模型上对之前的任务进行推理
    is_last_task_checkpoint = True
    
    if model_nums == 0:
        raise RuntimeError('there is no model')
    if model_nums < len(tasks):
        is_last_task_checkpoint = False
        print('任务没有完全训练完，现在只是输出已经训练完的模型的结果')
        tasks = tasks[:model_nums]
    ###########开始对刚刚微调完的任务进行评测################
    post_evaluation_filepath = os.path.join(args.save_path,'prediction')
    os.makedirs(post_evaluation_filepath,exist_ok=True)
    ###记得之后保存测试的文件时要按照*_*来进行保存
    evaluated = {
        os.path.basename(f).rsplit('_', 1)[-1].replace('.json', '')
        for f in glob.glob(f'{post_evaluation_filepath}/*_*.json')
    }

    need_eval_post_tasks = [task for task in tasks if task not in evaluated]
    print(f'--------------------------------post evaluate tasks:{need_eval_post_tasks}--------------------------------')

    for task_id, post_task in tqdm(list(enumerate(tasks)), desc="🔍 Post-task Evaluation", total=len(tasks)):
        if post_task not in need_eval_post_tasks:
            continue
        print(f'____________开始在任务{post_task}上评测____________')
        # post_model_path = os.path.join(args.model_paths,str(task_id+1))
        print(f'____________加载的模型是{args.model_paths}上评测____________')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_paths,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        )

        # default processer
        processor = AutoProcessor.from_pretrained(args.model_paths, use_fast=True)

        output_path = os.path.join(post_evaluation_filepath,f'{task_id+1}_{post_task}.json')
        
        evaluate_main(args,processor,model, post_task, output_path)
        del model
        release_memory()

        

