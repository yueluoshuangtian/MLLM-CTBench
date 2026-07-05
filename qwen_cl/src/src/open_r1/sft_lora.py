import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer,Qwen2_5_VLForConditionalGeneration
# 设置随机种子
def set_random_seed(seed):
    if seed is not None and seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_random_seed(1234)

# 加载模型和分词器
model_name = "/mnt/train_data/xjdu/model/Qwen2.5-VL-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# 配置 LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # 指定需要应用 LoRA 的模块
)
model = get_peft_model(model, lora_config)

def preprocess_function(examples):
    processed_examples = []
    for example in examples:
        messages = []
        for conversation in example["conversations"]:
            content = []
            if "images" in example:
                for image_path in example["images"]:
                    content.append({
                        "type": "image",
                        "image": image_path
                    })
            content.append({
                "type": "text",
                "text": conversation["content"]
            })
            messages.append({
                "role": conversation["role"],
                "content": content
            })
        processed_examples.append({"messages": messages})
    return processed_examples

# 加载数据集
dataset = load_dataset("json", data_files="/mnt/train_data/xjdu/LLaMA-Factory/data/maplm_train.json")

# 应用预处理函数
processed_dataset = dataset.map(preprocess_function, batched=False)

# 检查预处理后的数据
print(processed_dataset[0])

# 配置训练参数
training_args = TrainingArguments(
    output_dir="src/open-r1-multimodal/output/Qwen2.5-vL-3B-sft-maplm",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=3,
    learning_rate=5e-5,
    num_train_epochs=20,
    fp16=True,
    logging_steps=50,
    save_steps=2000,
    save_total_limit=5,
    deepspeed="src/open-r1-multimodal/local_scripts/zero2.json"
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

# 开始训练
trainer.train()