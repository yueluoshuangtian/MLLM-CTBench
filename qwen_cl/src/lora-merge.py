import os
import shutil
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from peft import PeftModel

# 定义设备
device = "cuda:6"

# 加载原始模型
base_model_path = "/mnt/train_data/xjdu/model/Qwen2.5-VL-3B-Instruct"  # 替换为原始模型路径
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)

# 加载 LoRA 权重
lora_path = "/mnt/train_data/xjdu/LLaMA-Factory/saves/qwen2.5_vl-3b/lora/sft/checkpoint-1500"  # 替换为 LoRA 模型的路径
model = PeftModel.from_pretrained(model, lora_path, device_map="auto", trust_remote_code=True)

# 合并模型
merged_model = model.merge_and_unload()

# 保存合并后的模型
save_path = "/public/home/houzhiyan/VLM-R1-main/src/open-r1-multimodal/output/Qwen2.5-VL-3B-sft-lora-merge"  # 替换为保存路径
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

shutil.copy(lora_path+'/preprocessor_config.json',save_path)
shutil.copy(lora_path+'/chat_template.json',save_path)
shutil.copy(lora_path+'/preprocessor_config.json',save_path)

print(f"合并后的模型已保存到 {save_path}")

# 验证合并后的模型
merged_model_path = save_path
merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
merged_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    merged_model_path,
    device_map=device,
    trust_remote_code=True
).eval()

# 测试推理
query = "Your test query here"  # 替换为测试问题
inputs = merged_tokenizer(query, return_tensors="pt").to(device)
outputs = merged_model.generate(**inputs)
response = merged_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Response:", response)