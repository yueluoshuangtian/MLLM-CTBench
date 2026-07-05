from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
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
import sys
project_root = "/public/home/houzhiyan/VLM-R1-main"
sys.path.append(project_root)
from RLCL_datasets.eval.eval import eval_seqft_tasks
import time
from vllm import LLM, SamplingParams
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
    "numglue": "/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test/numglue.json",
    "art": "/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test/art.json",
    "math": "/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test/math.json",
    "fomc": "/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test/fomc.json",
    "medical": "/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test/medical.json",
    "OCR": "/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test/OCR.json",
    "science": "/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test/science.json"
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
    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then {Question_propmt} in <answer> </answer> tags. Output the final answer in JSON format."
    messages = []

    for x in data:
        
        # pdb.set_trace()
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
                        "text": QUESTION_TEMPLATE.format(Question=x["problem"].split('prompt:\n')[0],Question_propmt = x["problem"].split('prompt:\n')[-1].lower().rstrip('.?'))
                    }
                ]
            }]
        else:
            message = [{
            "role": "user",
            "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question=x["problem"].split('prompt:\n')[0],Question_propmt = x["problem"].split('prompt:\n')[-1].lower().rstrip('.?'))}]
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
def evaluate_main(args,processor,model_path, task_val, output_path,task_name):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed
    )
    sampling_params = SamplingParams(
        temperature=0,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[],
    )
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

    for message in tqdm(messages, desc="推理的进度", unit=f"task:{task_name}"):
        prompt = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
        }
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)

        generated_text = outputs[0].outputs[0].text

        print(generated_text)
        all_outputs.append(generated_text)

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
    del llm
    gc.collect()
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', required=True, type=str) # Base model: "Qwen/Qwen2-VL-2B-Instruct"
    parser.add_argument("--image_root",default="/home/houzhiyan/dataset/images" ,type=str)
    parser.add_argument("--tasks", type=str)
    parser.add_argument("--sample_num", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--repetition_penalty",type=float,default=1.05)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument('--tensor_parallel_size',required=True,type=int)
    parser.add_argument('--dtype',required=True,type=str)
    return parser.parse_args()
import pdb
if __name__ == "__main__":
    
    args = parse_arguments()
    torch.manual_seed(args.seed)
    # pdb.set_trace()
    tasks = [cur_task.strip() for cur_task in args.tasks.split(',') if len(cur_task.strip())>0]
    model_nums = 0
    ##代表是否在最后的模型上对之前的任务进行推理
    is_last_task_checkpoint = True
    for d in glob.glob(f"{args.model_paths}/*"):
        if os.path.isdir(d) and os.path.basename(d).isdigit() and len(os.listdir(d)) > 0:
            model_nums = max(model_nums, int(os.path.basename(d)))
    if model_nums == 0:
        raise RuntimeError('there is no model')
    if model_nums < len(tasks):
        is_last_task_checkpoint = False
        print('任务没有完全训练完，现在只是输出已经训练完的模型的结果')
        tasks = tasks[:model_nums]
    ###########开始对刚刚微调完的任务进行评测################
    post_evaluation_filepath = os.path.join(args.model_paths,'prediction')
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
        post_model_path = os.path.join(args.model_paths,str(task_id+1))
        print(f'____________加载的模型是{post_model_path}上评测____________')
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        # post_model_path,
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        # device_map="cuda:0",
        # )

        # default processer
        processor = AutoProcessor.from_pretrained(post_model_path)

        output_path = os.path.join(post_evaluation_filepath,f'{task_id+1}_{post_task}.json')
        evaluate_main(args,processor,post_model_path, post_task, output_path,post_task)

        release_memory()

    ###########开始对在最后任务上微调完的模型进行评测################
    if not is_last_task_checkpoint:
        raise RuntimeError('由于没有在所有任务上微调完，所以没有进行last evaluate')

    last_evaluation_filepath = os.path.join(args.model_paths,'last_prediction')
    os.makedirs(last_evaluation_filepath,exist_ok=True)
    ###记得之后保存测试的文件时要按照*_*来进行保存
    last_evaluated = {
        os.path.basename(f).rsplit('_', 1)[-1].replace('.json', '')
        for f in glob.glob(f'{last_evaluation_filepath}/*_*.json')
    }

    need_eval_last_tasks = [task for task in tasks if task not in last_evaluated]
    print(f'--------------------------------last evaluate tasks:{need_eval_last_tasks}--------------------------------')
    last_model_path = os.path.join(args.model_paths,str(model_nums))
    print(f'____________加载的last模型是{last_model_path}上评测____________')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    last_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(last_model_path)
    for task_id, last_task in tqdm(list(enumerate(tasks)), desc="📊 Last-task Evaluation", total=len(tasks)):
        if last_task not in need_eval_last_tasks:
            continue
        if task_id + 1 == len(tasks):
            continue
        print(f'____________开始在第{task_id+1}个任务{last_task}上评测____________')

        output_path = os.path.join(last_evaluation_filepath,f'{task_id+1}_{last_task}.json')
        evaluate_main(args,processor,last_model_path, last_task, output_path,last_task)

    release_memory()
    ################对post和last的性能进行评测################
    # post_acc_metric = eval_all_tasks(args.model_paths,'post')
    # last_acc_metric = eval_all_tasks(args.model_paths,'last')
    # print(f"post_acc_metric is {post_acc_metric} \n")
    # print(f"last_acc_metric is {last_acc_metric}")

        

