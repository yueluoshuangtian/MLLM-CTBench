import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert import BertTokenizer, BertLMHeadModel

from llava.train.llava_trainer import maybe_zero_3

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class ExpandableProjection(nn.Module):
    def __init__(self, config, inst_input_size=768):
        super().__init__()
        self.num_experts = config.num_experts  # num_experts=task_id, to load weights correctly
        self.experts = nn.ModuleList([build_vision_projector(config) for i in range(self.num_experts)]) #使用nn.ModuleList创建一个包含多个专家网络的列表，每个专家网络由build_vision_projector(config)函数构建

        # expert keys trained used to retrieve the expert
        self.e_img_key = nn.Parameter(torch.zeros(config.mm_hidden_size, self.num_experts), requires_grad=True)
        self.e_inst_key = nn.Parameter(torch.zeros(inst_input_size, self.num_experts), requires_grad=True)
        nn.init.uniform_(self.e_img_key)
        nn.init.uniform_(self.e_inst_key)
        #当前任务的 ID，初始化为 -1。
        self.task_id = -1
        #键损失的权重系数。
        self.key_loss_lbd = 0.1
        self.loss = None

        # task encoder
        #从环境变量中获取 BERT 模型的路径，如果未设置则使用bert-base-uncased。
        load_dotenv()
        bert_model_path = os.getenv('BERT_BASE_UNCASED') or "bert-base-uncased"
        #初始化 BERT 分词器，并添加特殊标记[DEC]。
        self.max_txt_len = 128
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path, truncation_side="left")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        #加载 BERT 模型的配置，并修改编码器宽度和交叉注意力层的设置。
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

    def copy_from_proj(self, mm_projector):
        for expert in self.experts:
            for i, m in enumerate(expert):
                if isinstance(m, nn.Linear):
                    weight = maybe_zero_3(mm_projector[i].weight, ignore_status=True)
                    bias = maybe_zero_3(mm_projector[i].bias, ignore_status=True)
                    m.weight.data.copy_(weight.data)
                    m.bias.data.copy_(bias.data)

    def init_task_id_retrieve_acc(self):
        # used for computing task id retrieve accuracy when evaluation
        self.part_sizes = torch.zeros(self.num_experts)

    def cal_task_id_retrieve_acc(self):
        results = {f"task_id_retrieve/expert{i}": (s/self.part_sizes.sum()).item() for i, s in enumerate(self.part_sizes)}
        return results
    #该方法用于将对话列表convs转换为指令列表instructs。
    #使用正则表达式提取用户的指令。
    #在训练模式下，提取所有用户的指令；在评估模式下，只提取最后一个用户的指令。
    def conv2instr(self, convs):
        instructs = []
        instr_pattern = re.compile(r'(?<=USER: ).*?(?=ASSISTANT)')
        ans_pattern = re.compile(r'(?<=ASSISTANT: ).*?(?=</s>)')
        for conv in convs:
            if self.training:
                instruct = instr_pattern.findall(conv.replace("<image>", "").replace("\n", " "))
                instructs.extend(instruct)
            else:
                instruct = instr_pattern.findall(conv.replace("<image>", "").replace("\n", " "))[-1]
                instructs.append(instruct)
        return instructs

    #该方法用于将指令列表instructs输入到任务编码器中，得到指令的嵌入表示。
    #使用分词器对指令进行分词和编码。
    #将编码后的输入传递给 BERT 模型，得到最后一层的隐藏状态。
    #提取每个指令的第一个标记的隐藏状态作为指令的嵌入表示，并将其从计算图中分离出来。
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

    def forward(self, x, convs):
        # convert convs to inst_embeds
        inst_embeds = self.forward_task_encoder(self.conv2instr(convs), x.device)
        img_embeds = x[:, 0]
        x = x[:, 1:]

        if self.training:
            assert self.task_id is not None, "task id is required for training"
            n_K_img = F.normalize(self.e_img_key[:, self.task_id], dim=0)
            n_K_inst = F.normalize(self.e_inst_key[:, self.task_id], dim=0)
            q_img = F.normalize(img_embeds, dim=-1).detach()
            q_inst = F.normalize(inst_embeds, dim=-1).detach()
            loss = (1.0 - (q_img @ n_K_img)).sum() + (1.0 - (q_inst @ n_K_inst)).sum()

            # loss_pull to pull the expert close to the inputs of this task
            self.loss = loss * self.key_loss_lbd

            y_expert = self.experts[self.task_id](x) #使用当前任务 ID 对应的专家网络对输入数据进行投影，并返回输出。
            return y_expert 
        else:
            # # evaluate use the gt task id
            # y_expert = self.experts[self.task_id](x)

            n_K_img = F.normalize(self.e_img_key, dim=0)
            n_K_inst = F.normalize(self.e_inst_key, dim=0)
            q_img = F.normalize(img_embeds, dim=-1).detach()
            q_inst = F.normalize(inst_embeds, dim=-1).detach()
            cos_sim = (q_img @ n_K_img) + (q_inst @ n_K_inst)

            # cos_sim[:, 1:3] = -99

            # top_k = torch.topk(cos_sim[:, :self.task_id+1], 1, dim=1) # choose from learned experts
            top_k = torch.topk(cos_sim, 1, dim=1)
            task_id = int(top_k.indices[0])     # inference batch size is set to 1
            y_expert = self.experts[task_id](x)  #使用选择的专家网络对输入数据进行投影，并返回输出。
            self.part_sizes[task_id] += x.shape[0]  #更新该专家的样本数量统计信息。

            return y_expert


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

