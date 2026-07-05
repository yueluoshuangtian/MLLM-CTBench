import os
import copy
import json
import zlib
from PIL import Image
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import Dataset

import transformers
import tokenizers

from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)
def _subsample_list(data_list: list, ratio: float, seed: int, key: str):
    """
    对 data_list 做确定性下采样：同一 key（如 data_path）在多卡上抽到一致子集。
    ratio=1.0 不做任何事；ratio<1.0 仅保留子集。
    """
    if ratio is None or ratio >= 1.0:
        return data_list
    if ratio <= 0:
        raise ValueError(f"train_ratio must be in (0, 1], got {ratio}")

    n = len(data_list)
    k = max(1, int(n * ratio))

    # 用 key 派生 seed：同一文件 across ranks/重启 都一致；不同文件抽样不同
    derived = (seed ^ (zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)) & 0xFFFFFFFF
    rng = np.random.RandomState(derived)

    idx = rng.choice(n, k, replace=False)
    idx.sort()  # 保持样本原始顺序稳定（可选）

    return [data_list[i] for i in idx]


@dataclass
class DataArguments:
    # 训练数据目录（LLaVA conversations 格式，由 scripts/convert_qwen_to_llava.py 生成）；
    # 默认从环境变量 LLAVA_TRAIN_DIR 读取（配合 configs/paths.env）。
    data_dir: Optional[str] = field(default=os.environ.get("LLAVA_TRAIN_DIR", "data/llava_train"),
                                    metadata={"help": "Path to the json directory"})
    replay_replace_data_dir: Optional[str] = field(default=os.environ.get("LLAVA_REPLAY_DIR", "data/llava_train/changed_questions"),
                                    metadata={"help": "Path to the replay json directory"})

    tasks: Optional[str] = field(default=None,
                                 metadata={"help": "Path to the training data."})
    initial_tasks: Optional[str] = field(default="",
                                         metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=os.environ.get("IMAGE_ROOT", "data/images"))
    image_aspect_ratio: str = 'square'
    replay_ratio: float = 0.01
    non_duplicates:bool = True
    same_data_range:bool = False
    # ==========================
    # 控制当前任务主训练集的输入比例（默认 1.0 不影响旧逻辑）
    # ==========================
    train_ratio: float = 1.0         # 训练集使用比例：0.5/0.8/1.0
    train_ratio_seed: int = 42       # 下采样随机种子（多卡一致/可复现）
    
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>'
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(input_ids=input_ids, labels=targets)


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(convs=conversations, input_ids=input_ids, labels=targets)


def preprocess_plain(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)

    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    1) 拼接对话文本
    2) tokenization
    3) mask human tokens
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)

    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        replay_data_paths: list = None,
        fixed_datas: list = None,
        replay_replace_data_path: list = None
    ):
        super(LazySupervisedDataset, self).__init__()

        with open(data_path, "r", encoding="utf-8") as f:
            list_data_dict = json.load(f)

        # ==========================
        # NEW: 对“当前任务主训练集 data_path”按 train_ratio 下采样（默认 1.0 不变）
        # 说明：不影响 replay 的采样逻辑；多卡一致；可复现。
        # ==========================
        tr = getattr(data_args, "train_ratio", 1.0)
        trs = getattr(data_args, "train_ratio_seed", 42)
        list_data_dict = _subsample_list(
            list_data_dict,
            ratio=tr,
            seed=trs,
            key=os.path.abspath(data_path),
        )
        rank0_print(f"[TrainRatio] {os.path.basename(data_path)} keep={len(list_data_dict)} ratio={tr}")

        if replay_data_paths is not None:
            if fixed_datas is None:
                for path in replay_data_paths:
                    with open(path, "r", encoding="utf-8") as f:
                        list_data = json.load(f)
                    len_data = len(list_data)
                    len_selected = int(len_data * data_args.replay_ratio)
                    selected = np.random.choice(np.arange(len_data, dtype=np.int64), len_selected, replace=False)
                    list_data_dict.extend([list_data[i] for i in selected])
                    rank0_print(f"Loaded old dataset from {path}")

            else:
                for id, path in enumerate(replay_data_paths):
                    list_data = fixed_datas[path]
                    with open(replay_replace_data_path[id], "r", encoding="utf-8") as f:
                        replace_list_datas = json.load(f)
                    replace_map = {ex.get("id"): ex for ex in replace_list_datas}

                    if data_args.same_data_range:
                        len_data = len(list_data)
                        len_selected = int(0.5 * len_data)

                        selected = np.random.choice(np.arange(len_data, dtype=np.int64), len_selected, replace=False)
                        list_data_dict.extend([list_data[i] for i in selected])
                        selected_set = set(selected.tolist())
                        rank0_print(f"Loaded old dataset from {path},only {len_selected}")
                        missed = [i for i in range(len_data) if i not in selected_set]
                        added_replaced = 0
                        fallback_original = 0
                        for i in missed:
                            ex = list_data[i]
                            rid = ex.get("id")
                            rep = replace_map.get(rid)
                            if rep is not None:
                                list_data_dict.append(rep)
                                added_replaced += 1
                            else:
                                list_data_dict.append(ex)
                                fallback_original += 1

                        rank0_print(
                            f"Loaded old dataset from {path}, "
                            f"original kept={len_selected}, replaced_added={added_replaced}, fallback_original={fallback_original}"
                        )
                    else:
                        list_data_dict.extend(data for data in list_data)
                        rank0_print(f"Loaded old dataset from {path}")

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(sources, self.tokenizer, has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(
                convs=data_dict["convs"][0] if "convs" in data_dict else None,
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0]
            )

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        convs, input_ids, labels = tuple([instance[key] for instance in instances] for key in ("convs", "input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            convs=convs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def resolve_task_json_path(data_dir: str, task: str, split: str = "train") -> str:
    """
    兼容两种数据组织方式：
      1) 旧 LLaVA: data_dir/<task>.json
      2) 新 CoIN : data_dir/<task>/<split>.json  (例如 ScienceQA/train.json)
    """
    coin_path = os.path.join(data_dir, task, f"{split}.json")
    if os.path.isfile(coin_path):
        return coin_path

    legacy_path = os.path.join(data_dir, f"{task}.json")
    if os.path.isfile(legacy_path):
        return legacy_path

    raise FileNotFoundError(
        f"Cannot find json for task='{task}', split='{split}'. Tried:\n"
        f"  1) {coin_path}\n"
        f"  2) {legacy_path}\n"
    )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, rank: int) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    global local_rank
    local_rank = rank
    assert isinstance(data_args.tasks, str), "data_args.tasks should be a str."
    data_path = resolve_task_json_path(data_args.data_dir, data_args.tasks, split="train")
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_cl_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, task_list: str, rank: int) -> List[Dict]:
    """Make dataset and collator for continual learning fine-tuning."""
    global local_rank
    local_rank = rank

    tasks = [task.strip() for task in task_list.split(',') if len(task.strip()) > 0]
    data_paths = [resolve_task_json_path(data_args.data_dir, task, split="train") for task in tasks]
    replay_replace_data_path = [os.path.join(data_args.replay_replace_data_dir, f"{task}.json") for task in tasks]

    if data_args.initial_tasks is not None:
        initial_tasks = [task.strip() for task in data_args.initial_tasks.split(',') if len(task.strip()) > 0]
        initial_data_paths = [resolve_task_json_path(data_args.data_dir, task, split="train") for task in initial_tasks]
    else:
        initial_data_paths = []

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_modules = []
    rank0_print(f"Replay ratio: {data_args.replay_ratio}")

    if data_args.non_duplicates:
        for task_id, cur_data_path in enumerate(data_paths):
            rank0_print(f"Loading dataset from {cur_data_path}")
            train_dataset = LazySupervisedDataset(
                tokenizer=tokenizer,
                data_path=cur_data_path,
                data_args=data_args,
                replay_data_paths=data_paths[:task_id] if data_args.replay_ratio > 0 else None,
            )
            data_modules.append(dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator))
        return data_modules

    else:
        rank0_print('strat fixed_replay data collect')
        save_dir = os.path.join(data_args.data_dir, task_list)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{data_args.replay_ratio}.json")

        if os.path.isfile(save_path) and os.stat(save_path).st_size > 0:
            rank0_print(f"[Replay] Load fixed replay samples from {save_path}")
            with open(save_path, "r", encoding="utf-8") as f:
                fixed_datas: Dict[str, List[dict]] = json.load(f)
        else:
            fixed_datas: Dict[str, List[dict]] = {}
            for cur_data_path in data_paths:
                with open(cur_data_path, "r", encoding="utf-8") as f:
                    data_list = json.load(f)

                num_total = len(data_list)
                num_sample = int(num_total * data_args.replay_ratio)
                idx_sampled = np.random.choice(num_total, num_sample, replace=False)

                fixed_datas[cur_data_path] = [data_list[i] for i in idx_sampled]

                rank0_print(
                    f"[Replay] Sampled {num_sample}/{num_total} "
                    f"examples for {os.path.basename(cur_data_path)}"
                )

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(fixed_datas, f, ensure_ascii=False)
            rank0_print(f"[Replay] Fixed replay samples saved to {save_path}")

        for task_id, cur_data_path in enumerate(data_paths):
            rank0_print(f"Loading dataset from {cur_data_path}")

            train_dataset = LazySupervisedDataset(
                tokenizer=tokenizer,
                data_path=cur_data_path,
                data_args=data_args,
                replay_data_paths=data_paths[:task_id] if data_args.replay_ratio > 0 else None,
                fixed_datas=fixed_datas,
                replay_replace_data_path=replay_replace_data_path[:task_id] if data_args.replay_ratio > 0 else None,
            )

            data_modules.append(
                dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
            )

        return data_modules
                