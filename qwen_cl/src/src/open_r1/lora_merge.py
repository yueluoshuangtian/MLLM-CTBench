import os
import shutil
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration
from peft import PeftModel

import pdb
def merge_and_save_model(base_model_path, lora_path, save_path, device):
    """
    合并原始模型和LoRA权重，并保存合并后的模型。

    参数:
    base_model_path (str): 原始模型的路径。
    lora_path (str): LoRA权重的路径。
    save_path (str): 合并后模型保存的路径。
    device (str): 设备，默认为"cuda:7"。
    """
    # 如果路径不存在，则创建路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"路径 {save_path} 已创建")
    else:
        print(f"路径 {save_path} 已存在")
    # 加载原始模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    

    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, lora_path, device_map=device, trust_remote_code=True)

    # 合并模型
    merged_model = model.merge_and_unload()

    # 保存合并后的模型
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    def copy_or_ignore_if_same(src, dst):
        if os.path.abspath(src) == os.path.abspath(dst):
            # 如果源路径和目标路径相同，则不执行复制操作
            print("源路径和目标路径相同，无需复制。")
        else:
            # 如果路径不同，则执行复制操作
            shutil.copy(src, dst)
            print("文件已复制到目标位置。")
    copy_or_ignore_if_same(lora_path + '/preprocessor_config.json', save_path)
    copy_or_ignore_if_same(base_model_path + '/chat_template.json', save_path)

 

    print(f"合并后的模型已保存到 {save_path}")


def test_merged_model(merged_model_path, query, device):
    """
    测试合并后的模型。

    参数:
    merged_model_path (str): 合并后模型的路径。
    query (str): 测试问题。
    device (str): 设备，默认为"cuda:7"。
    """
    # 加载合并后的模型
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
    merged_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        merged_model_path,
        device_map=device,
        trust_remote_code=True
    ).eval()

    # 测试推理
    inputs = merged_tokenizer(query, return_tensors="pt").to(device)
    outputs = merged_model.generate(**inputs)
    response = merged_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Response:", response)
import glob
from tqdm import tqdm
import argparse
# 示例调用
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/public/share/iva_haiyun/Qwen2.5-VL-7B_sft-vqa")
    parser.add_argument("--lora_path", type=str, default="/public/share/iva_haiyun/Qwen2.5-VL-7B_GRPO-vqa")
    parser.add_argument("--save_path", type=str, default="/public/home/houzhiyan/VLM-R1-main/comment/Qwen2.5-VL-7B_GRPO-vqa")
    args = parser.parse_args()
    base_model_path = args.base_model_path
    lora_path = args.lora_path
    save_path = args.save_path
    
 
    device="cuda:0"
    
    merge_and_save_model(base_model_path,lora_path,save_path,device=device)
    #test_merged_model(save_path, query,device=device)