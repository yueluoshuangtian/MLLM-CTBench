import os
import json
import argparse
import warnings
import torch
from tqdm import tqdm
import shortuuid

import transformers
from transformers import BitsAndBytesConfig

from instruct_blip.data.dataset import create_test_loader
from instruct_blip.model.forward import replace_instruct_blip_forward, replace_instruct_blip_generate

import math

replace_instruct_blip_forward()
replace_instruct_blip_generate()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_pretrained_model(model_path, model_base, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'lora' in model_path.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
    if 'lora' in model_path.lower() and model_base is not None:
        lora_cfg_pretrained = transformers.AutoConfig.from_pretrained(model_path)
        print('Loading InstructBlip from base model...')
        model = transformers.InstructBlipForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        processor = transformers.InstructBlipProcessor.from_pretrained(model_base)

        print('Loading additional InstructBlip weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        # this may be language projection only
        print('Loading InstructBlip from base model...')
        processor = transformers.InstructBlipProcessor.from_pretrained(model_base)
        cfg_pretrained = transformers.AutoConfig.from_pretrained(model_path)
        model = transformers.InstructBlipForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        if hasattr(cfg_pretrained, 'num_experts'):
            from cl_learner.eproj import EprojLearner
            EprojLearner.initialize_model(model)
        print("Loading proj from model_path...")
        if os.path.isfile(os.path.join(model_path, 'mm_projector.bin')):
            module_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        elif os.path.isfile(os.path.join(model_path, 'qformer.bin')):
            module_weights = torch.load(os.path.join(model_path, 'qformer.bin'), map_location='cpu')
        else:
            raise FileNotFoundError
        module_weights = {k: v.to(torch.float16) for k, v in module_weights.items()}
        incompatible_keys = model.load_state_dict(module_weights, strict=False)
        print(incompatible_keys)
    else:
        processor = transformers.InstructBlipProcessor.from_pretrained(model_path)
        model = transformers.InstructBlipForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model



def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    processor, model = load_pretrained_model(model_path, args.model_base)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    data_loader = create_test_loader(questions, args.image_folder, processor)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if hasattr(model, "e_language_projection"):
        model.e_language_projection.init_task_id_retrieve_acc()

    model.eval()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        question_ids = batch.pop("question_ids")
        prompts = batch["prompts"]

        batch = {n: p.to(device="cuda", non_blocking=True) if isinstance(p, torch.Tensor) else p for n, p in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = model.generate(
                **batch,
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=False,
            )

        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)

        ans_id = shortuuid.uuid()
        for idx, prompt, output in zip(question_ids, prompts, outputs):
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": prompt,
                                       "text": output.strip(),
                                       "answer_id": ans_id,
                                       "model_id": "instruct_blip",
                                       "metadata": {}}) + "\n")

        if hasattr(model, "e_language_projection") and (i % 50 == 0 or i == len(questions) - 1):
            print(model.e_language_projection.cal_task_id_retrieve_acc())

        ans_file.flush()
    ans_file.close()

    if hasattr(model, "e_language_projection"):
        print(model.e_language_projection.cal_task_id_retrieve_acc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
