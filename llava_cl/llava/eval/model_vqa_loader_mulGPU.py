import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import torch.distributed as dist
from typing import Sequence, Dict
import transformers
from dataclasses import dataclass
import sys
sys.path.append('/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM')
from llava.model.builder import load_pretrained_model
from llava.train.train import rank0_print
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from transformers import StoppingCriteria
from PIL import Image
import math
import shutil
import deepspeed
directory_path = "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/new_eval_tool"
import sys
sys.path.insert(0, directory_path)
import evaluate_art_long_sentences,evaluate_FOMC,evaluate_math,evaluate_numglue,evaluate_science,evaluate_medical_long_sentences,evaluate_ocr

#推理用数据集的位置
question_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/numglue/NumGLUE_test_infer_use_reasoning.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/art/test_infer_use_reasoning.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/math/test_infer_use_reasoning.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/fomc/test_infer_use_reasoning.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/medical/test_infer_use_reasoning.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/ocr/test_infer_use_resoning.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/science/test_ann_infer_use_reasoning.json",
}

annotation_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/numglue/NumGLUE_test_answer.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/art/test_infer_answer.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/math/test_infer_answer.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/fomc/test_infer_answer.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/medical/test_infer_answer.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/ocr/test_infer_use_resoning.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/science/test_ann_infer_answer.json",
}
#question_iamge
image_folder_dict = {
    "numglue": "",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "fomc": "",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
}


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

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

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        inputs.update({"input_ids": input_ids,
                "convs": prompt})

        return inputs

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    #使用 zip(*batch) 可以将这个列表中的元组“转置”，使得你能分别获得所有样本的第一个元素、所有样本的第二个元素，以此类推。
    #适用于数据集（Dataset）的 __getitem__ 方法返回多个元素组成的元组
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes
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

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    #得到多个dict，每个dict由键值('question_id','prompt','input_ids','convs')
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=data_collator)
    return data_loader

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
import logging

def setup_logging(local_rank):
    """
    Set up logging to print errors only for rank 0, others go to a file or are suppressed.
    """
    if local_rank == 0:
        # For rank 0, log to console (stdout)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        # For other ranks, log to a file or suppress logging
        logging.basicConfig(filename=f'process_{local_rank}_error.log', level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


import random
def eval_model(args):
    local_rank = args.local_rank
    # Set up logging
    setup_logging(local_rank)
    # 确保每个进程绑定到唯一的 GPU
    torch.cuda.set_device(local_rank)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.config.use_cache = True

    answers_file = os.path.join(args.answers_file,args.task_name,args.task_order,"predictions",f"{args.task_id}.jsonl")
    rank0_print(f"开始进行{args.task_id}任务的推理,结果在{answers_file}中")
    evaluation_results_file = os.path.join(args.answers_file,args.task_name,args.task_order, "evaluations", f"{args.task_id}.jsonl")

    if (not os.path.isfile(answers_file)) or os.path.getsize(answers_file) == 0:
        acc = 0.
        #因为在准备训练数据的时候，通过batch生成
        model.config.tokenizer_padding_side = "left"
        tokenizer.padding_side = "left"

        question_file = question_file_dict[args.task_id]
        image_folder = image_folder_dict[args.task_id]
        num_chunks = args.num_chunks
        chunk_idx = local_rank
        answers_chunk_file = os.path.join(args.answers_file,args.task_name,args.task_order,"predictions",f"{args.task_id}",f"{num_chunks}-{chunk_idx}.jsonl")
        num_chunks = int(os.getenv('WORLD_SIZE', '1'))


        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        random.shuffle(questions) # 在主进程（rank 0）进行数据的打乱
        deepspeed.init_distributed()
        # 广播数据给所有进程
        questions_list = [None] * dist.get_world_size()  # 用于接收广播的容器
        if local_rank == 0:
            questions_list[0] = questions
        dist.broadcast_object_list(questions_list, src=0)
        questions = questions_list[0]  # 所有进程上的 questions 都是一样的
        questions = get_chunk(questions, num_chunks, chunk_idx)
        answers_chunk_file = os.path.expanduser(answers_chunk_file)

        os.makedirs(os.path.dirname(answers_chunk_file), exist_ok=True)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        os.makedirs(os.path.dirname(evaluation_results_file),exist_ok=True)
        ans_file = open(answers_chunk_file, "w")

        if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
            args.conv_mode = args.conv_mode + '_mmtag'
            rank0_print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

        # conv = conv_templates[args.conv_mode].copy()
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]

        data_loader = create_data_loader(questions, image_folder, tokenizer, image_processor, model.config)


        
        model.gradient_checkpointing_disable()
        model.eval()

        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), miniters=len(data_loader)//10, disable=(local_rank != 0)):
            input_ids = batch["input_ids"]
            convs = batch["convs"]
            attn_mask = batch["attention_mask"]

            question_ids = batch["question_ids"]
            prompts = batch["prompts"]

            if 'images' in batch:
                image_tensor = batch["images"].to(device='cuda', non_blocking=True, dtype=torch.bfloat16)
            else:
                image_tensor = None

            input_ids = input_ids.to(device='cuda', non_blocking=True)
            attn_mask = attn_mask.to(device='cuda', non_blocking=True)
            # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attn_mask,
                    images=image_tensor,
                    convs=convs,
                    do_sample=False,
                    # temperature=1.0 ,
                    temperature = args.temperature,
                    top_p=1.0,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    use_cache=False
                    # stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                rank0_print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

            ans_id = shortuuid.uuid()
            for idx, prompt, output in zip(question_ids, prompts, outputs):
                ans_file.write(json.dumps({"question_id": idx,
                                            "prompt": prompt,
                                            "text": output.strip(),
                                            "answer_id": ans_id,
                                            "model_id": str(args.task_id),
                                            "metadata": {}}) + "\n")

            if hasattr(model.get_model(), "e_mm_projector") and (i % 50 == 0 or i == len(questions) - 1):
                rank0_print(model.get_model().e_mm_projector.cal_task_id_retrieve_acc())

            ans_file.flush()
        
        ans_file.close()
        dist.barrier()  # wait for all chunks are completed

        if local_rank == 0:
            ans_file = open(answers_file, "w")
            for i in range(num_chunks):
                for q in open(
                        os.path.join(args.answers_file,args.task_name,args.task_order,"predictions",f"{args.task_id}", f"{num_chunks}-{i}.jsonl")):
                    ans_file.write(q)
                    ans_file.flush()
            ans_file.close()

    else:
        rank0_print( f"推理答案{answers_file}已存在")

    if local_rank == 0:
        if os.path.isdir(os.path.dirname(evaluation_results_file)):
            shutil.rmtree(evaluation_results_file)
        if args.task_id == "art":
            acc = evaluate_art_long_sentences.eval_results(annotation_file_dict[args.task_id], answers_file,evaluation_results_file,is_reasoning=True)
        elif args.task_id == "numglue":
            acc = evaluate_numglue.eval_results(annotation_file_dict[args.task_id], answers_file,evaluation_results_file,is_reasoning=True)
        elif args.task_id == "math":
            acc = evaluate_math.eval_results(annotation_file_dict[args.task_id], answers_file,evaluation_results_file,is_reasoning=True)
        elif args.task_id == "fomc":
            acc = evaluate_FOMC.eval_results(annotation_file_dict[args.task_id], answers_file,evaluation_results_file,is_reasoning=True)
        elif args.task_id == "medical":
            acc = evaluate_medical_long_sentences.eval_results(annotation_file_dict[args.task_id], answers_file,evaluation_results_file,is_reasoning=True)
        elif args.task_id == "OCR":
            acc = evaluate_ocr.eval_results(annotation_file_dict[args.task_id], answers_file,evaluation_results_file,is_reasoning=True)
        elif args.task_id == "science":
            acc = evaluate_science.eval_results(annotation_file_dict[args.task_id], answers_file,evaluation_results_file,is_reasoning=True)
        else:
            raise ValueError(f"Unknown task : {args.task_id}")

    dist.barrier()

    model.config.tokenizer_padding_side = "right"
    tokenizer.padding_side = "right"
    rank0_print( f"任务{args.task_id}推理结果{answers_file}的正确率是{acc}")
    return acc

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    #task_ids = ["numglue","art","math","fomc","medical","OCR","science"]
    parser.add_argument("--task-id",type=str,default='')
    #task_name包括baseline\ewc、lwf等等
    parser.add_argument("--task-name",type=str,default='')
    #微调的顺序：sequent、reverse
    parser.add_argument("--task-order",type=str,default='')

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="")
    # parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    #此处的answer_file是整个微调任务的存放位置（例如顺序微调的所有结果）
    parser.add_argument("--answers-file", type=str, default="answer")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument('--local_rank', type=int, default=None, help='Local rank for distributed training')
    args = parser.parse_args()

    eval_model(args)
