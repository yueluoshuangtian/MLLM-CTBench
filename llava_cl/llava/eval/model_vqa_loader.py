import argparse
import torch
import os
import json
from tqdm import tqdm
from typing import Sequence, Dict
from dataclasses import dataclass
import shortuuid

import transformers
from transformers import StoppingCriteria

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from llava.constants import IGNORE_INDEX

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["text"]

        inputs = {"question_id": line["question_id"],
                  "prompt": line["text"]}

        if 'image' in line:
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            image_file = line["image"]
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            inputs["image"] = process_images([image], self.image_processor, self.model_config)[0]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        inputs.update({"input_ids": input_ids,
                       "convs": prompt})

        return inputs

    def __len__(self):
        return len(self.questions)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.
    有几个问题要注意：1、由于batch！=1，所以需要进行padding，同时也要让model和tokenizer进行padding
    """

    tokenizer: transformers.PreTrainedTokenizer #Python 的类型注解，用于表明类属性 tokenizer 应该是一个 transformers.PreTrainedTokenizer 类型的对象。

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        convs, input_ids, question_ids, prompts = tuple([instance[key] for instance in instances]
                                                        for key in ("convs", "input_ids", "question_id", "prompt"))
        """给定一个由一维或多维张量组成的列表，这些张量的第一个维度代表序列长度，pad_sequence 会在较短的序列后面（或通过特定操作可实现左侧填充）填充指定的值，使得所有序列具有相同的长度。

常用参数：

sequences：一个张量列表，每个张量代表一个序列。
batch_first：布尔值。如果为 True，返回的张量的形状将是 [batch_size, max_length, ...]；如果为 False，形状为 [max_length, batch_size, ...]。
padding_value：填充值，用于填充较短的序列部分。"""
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [i.flip(dims=[0]) for i in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id).flip(dims=[1])  #self.tokenizer.pad_token_id指的时分词器预定义的pad_token
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        batch = dict(
            convs=convs,
            question_ids=question_ids,
            prompts=prompts,
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)
            flag = True
            for output in outputs:
                for keyword in self.keywords:
                    if keyword not in output:
                        flag = False
                        return False
            return flag


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    #每次从数据集中按照batch_size数量进行采样
    # data_loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    data_loader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    model_name = "llava"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)

    if hasattr(model.get_model(), "e_mm_projector"):
        model.get_model().e_mm_projector.init_task_id_retrieve_acc()

    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    model.eval()
    for i, (batch, line) in tqdm(enumerate(zip(data_loader, questions)), total=len(questions)):
        input_ids = batch["input_ids"]
        image_tensor = batch["images"]
        convs = batch["convs"]
        attn_mask = batch["attention_mask"]

        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attn_mask = attn_mask.to(device='cuda', non_blocking=True)
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attn_mask,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                convs=convs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=False,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")


        if i % 50 == 0 and hasattr(model.get_model(), "e_mm_projector"):
            print(model.get_model().e_mm_projector.cal_task_id_retrieve_acc())

        # ans_file.flush()
    ans_file.close()

    if hasattr(model.get_model(), "e_mm_projector"):
        print(model.get_model().e_mm_projector.cal_task_id_retrieve_acc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
