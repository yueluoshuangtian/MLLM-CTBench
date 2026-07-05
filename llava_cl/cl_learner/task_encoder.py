import os
import re
import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert import BertTokenizer, BertLMHeadModel
from dotenv import load_dotenv


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class TaskEncoder:

    cur_task_embeds = None
    task_embeds = None
    task_sim = dict()

    def __init__(self, vision_tower, rank, output_dir):
        global local_rank
        local_rank = rank

        self.num_data_for_ipt = 1e3
        self.batch_size = 16
        self.device = torch.device("cuda", local_rank)
        self.save_file = os.path.join(output_dir, "task_sim.bin")

        self.vision_encoder = vision_tower
        self.vision_encoder.to(self.device)
        self.vision_encoder.eval()

        load_dotenv()
        bert_model_path = os.getenv('BERT_BASE_UNCASED') or "bert-base-uncased"

        self.max_txt_len = 128
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path, truncation_side="left")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        encoder_config = BertConfig.from_pretrained(bert_model_path)
        encoder_config.encoder_width = 1408
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = False
        self.task_encoder = BertLMHeadModel.from_pretrained(bert_model_path, config=encoder_config)

        self.task_encoder.to(self.device)
        self.task_encoder.eval()

    def save(self):
        rank0_print("Saving task sim states...")
        if local_rank <= 0:
            os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
            save_states = {
                "task_sim": self.task_sim,
                "task_embeds": self.task_embeds
            }
            torch.save(save_states, self.save_file)
        dist.barrier()

    def load(self):
        if os.path.isfile(self.save_file) and self.task_embeds is None:
            rank0_print(f"Loading task_embeds from {self.save_file}")
            ckpt = torch.load(self.save_file, map_location="cpu")
            self.task_embeds = ckpt["task_embeds"]
            self.task_sim = ckpt["task_sim"]
            self.task_embeds = {n: p.to(self.device) for n, p in self.task_embeds.items()}
            self.task_sim = {n: p.to(self.device) for n, p in self.task_sim.items()}

    def cal_task_sim(self, num_seen_tasks, train_dataset, data_collator, **kwargs):
        self.load()
        load_num_seen_tasks = list(self.task_sim.values())[0].shape[0] if len(self.task_sim) > 0 else 0

        if num_seen_tasks > 0 and num_seen_tasks == load_num_seen_tasks:
            task_sim = 1
            for m, sim in self.task_sim.items():
                task_sim *= sim
                rank0_print(f"{m} Task similarity: {sim.cpu().float().numpy()}")
            rank0_print(f"Overall task similarity: {task_sim.cpu().float().numpy()}")

        elif num_seen_tasks < load_num_seen_tasks:
            task_sim = None

        else:
            self.cur_task_embeds = self.get_task_emb(train_dataset, data_collator)
            if self.task_embeds is not None:
                task_sim = 1
                for m, task_embed in self.cur_task_embeds.items():
                    temp_task_embeds = torch.vstack((task_embed, self.task_embeds[m]))
                    sim_mat = (temp_task_embeds @ temp_task_embeds.T).triu(diagonal=1)
                    sim = sim_mat.masked_select(sim_mat != 0)
                    # standardize to align different metrics
                    if sim.size(0) > 1:
                        sim = torch.sigmoid((sim - sim.mean()) / sim.std())
                    sim = sim[:temp_task_embeds.size(0)-1]
                    task_sim *= sim
                    self.task_sim[m] = sim
                    rank0_print(f"{m} Task similarity: {sim.cpu().float().numpy()}")
                rank0_print(f"Overall task similarity: {task_sim.cpu().float().numpy()}")
                for m, embed in self.cur_task_embeds.items():
                    self.task_embeds[m] = torch.vstack((self.task_embeds[m], embed))
            else:
                self.task_embeds = {p: v.unsqueeze(0) for p, v in self.cur_task_embeds.items()}
                task_sim = None
            self.save()

        return task_sim

    @torch.inference_mode()
    def get_task_emb(self, train_dataset, data_collator):
        #间隔抽取token imbedding
        if len(train_dataset) > self.num_data_for_ipt:
            train_dataset = copy.deepcopy(train_dataset)    # do not change the dataset for training
            interval = int(len(train_dataset) // self.num_data_for_ipt)
            train_dataset.list_data_dict = train_dataset.list_data_dict[::interval]

        world_size = int(os.getenv('WORLD_SIZE', '1'))
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=world_size,
                                           rank=local_rank)
        dataloader = DataLoader(train_dataset,
                                collate_fn=data_collator,
                                sampler=train_sampler,
                                batch_size=self.batch_size)
        rank0_print("Computing task embeds...")
        progress_bar = tqdm(total=len(dataloader), leave=True, disable=(local_rank != 0))

        img_embed = 0
        inst_embed = 0
        ans_embed = 0
        for sample in dataloader:

            if 'images' in sample:
                images = sample['images'].to(self.device)
                img_embed += self.vision_encoder(images)[:, 0].detach().mean(0)
                convs = sample['convs']
                inst, ans = self.conv2instr(convs)
            elif 'pixel_values' in sample:
                pixel_values = sample['pixel_values'].to(self.device)
                img_embed += self.vision_encoder(pixel_values=pixel_values)[0][:, 0, :].detach().mean(0)
                inst, ans = sample['prompts'], sample['answers']
            else:
                raise ValueError(f"Unexpected sample: {sample}")

            inst_embed += self.forward_task_encoder(inst).detach().mean(0)
            ans_embed += self.forward_task_encoder(ans).detach().mean(0)

            progress_bar.update(1)

        len_loader = torch.tensor(len(dataloader)).float().to(self.device)

        dist.all_reduce(len_loader)
        dist.all_reduce(img_embed)
        dist.all_reduce(inst_embed)
        dist.all_reduce(ans_embed)

        mean_img_embed = F.normalize(img_embed / len_loader, dim=0)
        mean_inst_embed = F.normalize(inst_embed / len_loader, dim=0)
        mean_ans_embed = F.normalize(ans_embed / len_loader, dim=0)

        return {
            "img": mean_img_embed,
            "inst": mean_inst_embed,
            "ans": mean_ans_embed
        }

    def forward_task_encoder(self, instructs):
        tokens = self.tokenizer(
            instructs,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        instruct_output = self.task_encoder.bert(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
            return_dict=True
        )
        instruct_embeds = instruct_output.last_hidden_state[:, 0, :].detach()
        return instruct_embeds

    def conv2instr(self, convs):
        all_inst = []
        all_ans = []
        inst_pattern = re.compile(r'(?<=USER: ).*?(?=ASSISTANT)')
        ans_pattern = re.compile(r'(?<=ASSISTANT: ).*?(?=</s>)')
        for conv in convs:
            inst = inst_pattern.findall(conv.replace("<image>", "").replace("\n", " "))
            ans = ans_pattern.findall(conv.replace("<image>", "").replace("\n", " "))
            all_inst.extend(inst)
            all_ans.extend(ans)

        return all_inst, all_ans
