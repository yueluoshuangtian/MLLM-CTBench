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
import pdb
from dataclasses import dataclass, field
from typing import Optional
from transfer_to_critic import qwen_trans_to_comment

def str_to_bool(value):
    """
    Convert a string to a boolean value.
    Accepted values: 'true', 'false', '1', '0'
    """
    if value.lower() in ['true', '1']:
        return True
    elif value.lower() in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_score(response:str):
    pattern = r"Final Score for Response (1|2):\s*(\d+\.?\d*)\s*"
    matches = re.search(pattern, response)
    return float(matches.group(2))
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str) # Base model: "Qwen/Qwen2-VL-2B-Instruct"
    parser.add_argument('--bs', default=32, type=int) # Batch size: reduce it if GPU OOM
    parser.add_argument('--output_dir', default="results", type=str)
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--test_datasets_task", nargs="+", type=str)
    parser.add_argument("--sample_num", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--is_vqa", default="True", type=str_to_bool)
    
    
    return parser.parse_args()
import glob
if __name__ == "__main__":
    args = parse_arguments()
    is_vqa = args.is_vqa
    MODEL_PATH = args.model_path
    OUTPUT_PATH = args.output_dir

    BSZ=args.bs
    DATA_ROOT = args.data_root

    TEST_DATASETS = args.test_datasets_task
    IMAGE_ROOT = args.image_root
    # TEST_DATASETS = ['refgta_subsample']
    # IMAGE_ROOT = "path/to/refgta"
    max_new_tokens = args.max_new_tokens
    random.seed(42)
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    
    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    sample_num = args.sample_num
    #TEST_DATASETS 测试文件的名称
    #DATA_ROOT root 文件夹的名称
    
    for ds in TEST_DATASETS:
        print(f"Processing {ds}...")
        
        
        ds_paths = glob.glob(f"{DATA_ROOT}/*{ds}*.json",recursive=True)
        print(f"任务是",ds)
        print(f"得到的任务路线是",ds_paths)
        if  len(ds_paths) == 1:
            print("文件正确")
            ds_path = ds_paths[0]
        if "qwen" in MODEL_PATH.lower():
            data = qwen_trans_to_comment(ds_path)

        #下面需要修改QUESTION_TEMPLATE和message

        data = data[:sample_num]
        messages = []

        for x in data:
            QUESTION_TEMPLATE = "{Question}.**Task:{process_Question_propmt} First output the thinking process in <think> </think> tags and then {answer_Question_propmt} in <answer> </answer> tags. Output the final answer in JSON format."
            question = x["problem"].split("**Task:")[0].strip()
            process_prompt = x['problem'].split("**Task:")[-1].replace(",and provide an overall score.",".")
            answer_prompt = "Provide the overall score in the format: \"Final Score for Response 1: score\""

            
            # pdb.set_trace()
            if "image" in x.keys():
                image_path = os.path.join(IMAGE_ROOT, x['image'])
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
                            "text": QUESTION_TEMPLATE.format(Question = question,process_Question_propmt = process_prompt,answer_Question_propmt = answer_prompt)
                        }
                    ]
                }]
            else:
                message = [{
                "role": "user",
                "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question = question,process_Question_propmt = process_prompt,answer_Question_propmt = answer_prompt)}]
                },
                {
                    
                        "role": "assistant",
                        "content": x['solution'],
                       
                }
            ]
    
            messages.append(message)
        
        all_outputs = []  # List to store all answers
        #BSG batch_size
        # Process data
        for i in tqdm(range(0, len(messages), BSZ)):
            batch_messages = messages[i:i + BSZ]
   
            # Preparation for inferences
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            # pdb.set_trace()
            if is_vqa:
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

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            all_outputs.extend(batch_output_text)
            # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

        final_output = []
        all_eva_score = 0.0
        nums = len(data)
        wrong_ids = []

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            try:
                score = get_score(original_output)
            except:
                wrong_ids.append(input_example['question_id'])
                score = 0
                nums += -1
            all_eva_score += score
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
                # 'correct': correct
            }
            final_output.append(result)
        
        

 
        # Save results to a JSON file
        output_path = os.path.join(OUTPUT_PATH,f'{ds}.json')
        os.makedirs(OUTPUT_PATH,exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "num":nums,
                "wrong_ids":wrong_ids,
                'results': final_output
            }, f, indent=2)
        average_score = all_eva_score/nums
        with open(output_path, "w") as f:
            json.dump({
                "num":nums,
                "wrong_ids":wrong_ids,
                'eval_score': f"{average_score:.2f}",
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")






