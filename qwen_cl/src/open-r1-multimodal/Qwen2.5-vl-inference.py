from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import infer_auto_device_map
# default: Load the model on the available device(s)
import torch
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/public/home/houzhiyan/Qwen2.5-VL-3B-Instruct", torch_dtype="auto"#, device_map="auto"
)

model = model.to("cuda")
torch.cuda.empty_cache()
# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 640*28*28
processor = AutoProcessor.from_pretrained("/public/home/houzhiyan/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
#'/data2/xjdu/model/Qwen2.5-VL-3B-Instruct'
#Question="Please provide the bounding box coordinate of the region this sentence describes:  狗."
Question="一条狗"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/houzhiyan/dataset/images/art_vqa_datasets/AQUA/train/train_author_images/00066-ark1.jpg",
            },
            {"type": "text", "text": f"{Question}First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.Output the final answer in JSON format."},

        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    padding_side="left",
    add_special_tokens=False,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)