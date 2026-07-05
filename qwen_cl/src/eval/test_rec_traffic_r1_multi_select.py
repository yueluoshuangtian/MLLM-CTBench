from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random

# steps = 500
# print("Steps: ", steps)
# MODEL_PATH=f"/data/xjdu/VLM-R1-main/src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-REC-lora/checkpoint-{steps}" 

def calculate_reward(gt_answer, content_answer):
    """
    计算奖励：
    - 解析 gt_answer 和 content_answer 中的选项标识符和文本。
    - 如果选项标识符匹配，则奖励为 1.0。
    - 如果文本匹配，则奖励为 1.0。
    - 比较时不区分大小写。
    """
    # 定义选项标识符和文本的正则表达式
    option_pattern = r"([^\s]+)[：:](.+)" #r"([^\s]+):(.+)"
    
    # 解析 gt_answer
    gt_match = re.match(option_pattern, gt_answer.strip(), re.IGNORECASE)
    if gt_match:
        gt_option_id = gt_match.group(1).strip()
        gt_option_text = gt_match.group(2).strip()
    else:
        # 如果 gt_answer 不符合格式，直接将整个文本作为匹配内容
        gt_option_id = gt_answer.strip()
        gt_option_text = gt_answer.strip()
    
    # 解析 content_answer
    content_match = re.match(option_pattern, content_answer.strip(), re.IGNORECASE)
    if content_match:
        content_option_id = content_match.group(1).strip()
        content_option_text = content_match.group(2).strip()
    else:
        # 如果 content_answer 不符合格式，直接将整个文本作为匹配内容
        content_option_id = content_answer.strip()
        content_option_text = content_answer.strip()
    
    # 比较选项标识符和文本
    if (gt_option_id in content_option_id) or \
       (gt_option_text in content_option_text):
        reward = 1.0
    else:
        reward = 0.0
    
    return reward
def answer_(content, solution):
    """
    如果模型回答的answer部分包含GT的选项号码或选项内容，则奖励为1.0，否则为0.0。
    
    参数：
    - completions: 模型生成的回答列表，每个元素是一个包含回答内容的字典。
    - solution gt_answers: 正确答案列表，每个元素可以是选项号码（字符串形式）或选项内容。
    """
    
    rewards = 0
    #import pdb;pdb.set_trace()
    answer_tag_pattern = r'<answer>(.*?)</answer>'    

    try:
        
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        gt_answer_match = re.search(answer_tag_pattern, solution, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            gt_answer = gt_answer_match.group(1).strip() 
            reward = calculate_reward(gt_answer,content_answer)
        return reward,content_answer
        
    except Exception as e:
        print(f"Error processing answer: {e}")
        return 0,None  # 继续处理下一个答案
        
    
        
                
    
    return rewards
def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            x1, y1, x2, y2 = bbox
            return bbox, False
    return [0, 0, 0, 0], False

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union


MODEL_PATH= "/public/home/houzhiyan/VLM-R1-main/src/open-r1-multimodal/output/Qwen2.5-VL-3B-GRPO-REC-lora-20epoch-merge"
#DATASET="mapvl"
DATASET="test"

#OUTPUT_PATH="./logs/rec_results_{DATASET}_qwen2_5vl_3b_instruct_r1_{STEPS}.json"
model_name = os.path.basename(MODEL_PATH)
OUTPUT_PATH=f"./logs/rec_results_{model_name}_{DATASET}.json"
output_path = OUTPUT_PATH
BSZ=20
device ="cuda:0"
random.seed(42)
sample_num = 3
#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

#ds_path = "/mnt/train_data/xjdu/MAPLM-V2/maplm_v0.1/filtered_problems_test.json"
ds_path = "/public/home/houzhiyan/VLM-R1-main/data/test-multi-select/test-multi-select.json"


data = json.load(open(ds_path, "r"))
random.shuffle(data)
# QUESTION_PROMPT = "Please carefully observe the image, which is composed of three sub-images taken from different perspectives, with each sub-image labeled with its corresponding perspective in the upper left corner. These three sub-images are, in order: the left rear view, the front view, and the right rear view. Based on the following questions, please select all correct options.\n"
# QUESTION_TEMPLATE = QUESTION_PROMPT+"{Question} First, output the thinking process within the <think> and </think> tags. Then, select the correct answer from the provided options and output the final answer within the <answer> and </answer> tags. The final answer should include the correct option and its corresponding text content."

QUESTION_PROMPT = "请仔细观察图像，该图像由三个从不同角度拍摄的子图像组成，每个子图像的左上角都标有其对应的视角。这三个子图像依次为：左后视图、正视图和右后视图。根据以下问题，请选择所有正确的选项。\n"
QUESTION_TEMPLATE = QUESTION_PROMPT + "{Question} 首先，在 <think> 和 </think> 标签内输出思考过程。然后，从提供的选项中选择所有正确的答案，并在 <answer> 和 </answer> 标签内输出最终答案。最终答案应仅包含正确的选项及其对应的文本内容。"

#QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
data = data[:sample_num]
messages = []

for x in data:
    #import pdb;pdb.set_trace()
    image_path = x['image_path']
    message = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{image_path}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=x['problem'])
            }
        ]
    }]
    messages.append(message)

all_outputs = []  # List to store all answers

# Process data
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    all_outputs.extend(batch_output_text)
    # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

final_output = []
correct_number = 0

for input_example, model_output in zip(data, all_outputs):
    correct = 0
    original_output = model_output
    ground_truth = input_example['solution']
    correct,model_answer = answer_(original_output,ground_truth)
    # Count correct answers
    
    correct_number += correct
    
    # Create a result dictionary for this example
    result = {
        'question': input_example['problem'],
        'ground_truth': ground_truth,
        'model_output': original_output,
        'extracted_answer': model_answer,
        'correct': correct
    }
    final_output.append(result)
    print('model_output',model_answer)

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy of mapvl: {accuracy:.2f}%")

# Save results to a JSON file

output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_path, "w",encoding="utf-8") as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f, indent=2, ensure_ascii=False)

print(f"Results saved to {output_path}")
print("-"*100)


