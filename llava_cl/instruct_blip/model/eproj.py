import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert import BertTokenizer, BertLMHeadModel

from llava.train.llava_trainer import maybe_zero_3

class ExpandableProjection(nn.Module):
    def __init__(self, config, inst_input_size=768):
        super().__init__()
        self.num_experts = config.num_experts   # num_experts=task_id, to load weights correctly
        self.experts = nn.ModuleList([nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
                                      for i in range(self.num_experts)])

        # expert keys trained used to retrieve the expert
        self.e_img_key = nn.Parameter(torch.zeros(config.vision_config.hidden_size, self.num_experts), requires_grad=True)
        self.e_inst_key = nn.Parameter(torch.zeros(inst_input_size, self.num_experts), requires_grad=True)
        nn.init.uniform_(self.e_img_key)
        nn.init.uniform_(self.e_inst_key)

        self.task_id = -1

        self.key_loss_lbd = 0.1
        self.loss = None

        # task encoder
        load_dotenv()
        bert_model_path = os.getenv('BERT_BASE_UNCASED') or "bert-base-uncased"

        self.max_txt_len = 128
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path, truncation_side="left")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        encoder_config = BertConfig.from_pretrained(bert_model_path)
        encoder_config.encoder_width = 1408
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = False
        self.task_encoder = BertLMHeadModel.from_pretrained(
            bert_model_path, config=encoder_config
        )
        self.task_encoder.eval()
        for name, param in self.task_encoder.named_parameters():
            param.requires_grad = False

    def set_task_id(self, task_id):
        self.task_id = task_id

    def copy_from_proj(self, language_projector):
        for expert in self.experts:
            weight = maybe_zero_3(language_projector.weight, ignore_status=True)
            bias = maybe_zero_3(language_projector.bias, ignore_status=True)
            expert.weight.data.copy_(weight.data)
            expert.bias.data.copy_(bias.data)

    def init_task_id_retrieve_acc(self):
        # used for computing task id retrieve accuracy when evaluation
        self.part_sizes = torch.zeros(self.num_experts)

    def cal_task_id_retrieve_acc(self):
        results = {f"task_id_retrieve/expert{i}": (s/self.part_sizes.sum()).item() for i, s in enumerate(self.part_sizes)}
        return results

    def forward_task_encoder(self, instructs, device):
        tokens = self.tokenizer(
            instructs,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        instruct_output = self.task_encoder.bert(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
            return_dict=True
        )
        instruct_embeds = instruct_output.last_hidden_state[:, 0, :].detach()
        return instruct_embeds

    def forward(self, x, img_embeds, prompts):
        inst_embeds = self.forward_task_encoder(prompts, x.device)

        if self.training:
            assert self.task_id is not None, "task id is required for training"
            n_K_img = F.normalize(self.e_img_key[:, self.task_id], dim=0)
            n_K_inst = F.normalize(self.e_inst_key[:, self.task_id], dim=0)
            q_img = F.normalize(img_embeds, dim=-1).detach()
            q_inst = F.normalize(inst_embeds, dim=-1).detach()
            loss = (1.0 - (q_img @ n_K_img)).sum() + (1.0 - (q_inst @ n_K_inst)).sum()

            # loss_pull to pull the expert close to the inputs of this task
            self.loss = loss * self.key_loss_lbd

            y_expert = self.experts[self.task_id](x)
            return y_expert

        else:
            n_K_img = F.normalize(self.e_img_key, dim=0)
            n_K_inst = F.normalize(self.e_inst_key, dim=0)
            q_img = F.normalize(img_embeds, dim=-1).detach()
            q_inst = F.normalize(inst_embeds, dim=-1).detach()
            cos_sim = (q_img @ n_K_img) + (q_inst @ n_K_inst)

            top_k = torch.topk(cos_sim, 1, dim=1)
            task_id = int(top_k.indices[0])     # inference batch size is set to 1
            y_expert = self.experts[task_id](x)
            self.part_sizes[task_id] += x.shape[0]

            return y_expert
