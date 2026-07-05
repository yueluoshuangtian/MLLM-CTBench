# coding=utf-8                      # 指定源码文件的编码为 UTF-8，保证中文/特殊字符正确解析。
# Copyright 2023-present ...        # 版权与许可声明（Apache-2.0）。
# Licensed under the Apache ...     # 说明此文件受 Apache License 2.0 约束。
# 省略若干许可文本...

from __future__ import annotations    # 启用“延迟注解”特性：类型注解在运行时以字符串形式存在，避免前置定义/循环依赖问题。

from typing import TYPE_CHECKING, Any, Dict   # 类型系统相关：TYPE_CHECKING 用于静态类型检查时的条件导入；Any/Dict 是通用类型别名。

from .peft_model import (                    # 从当前包的 peft_model 子模块中导入一组 PEFT 模型包装类。
    PeftModel,                               # 基础通用的 PEFT 包装类（兜底）。
    PeftModelForCausalLM,                    # 面向 Causal LM 任务的包装类。
    PeftModelForFeatureExtraction,           # 面向特征提取任务的包装类。
    PeftModelForQuestionAnswering,           # 面向问答任务的包装类。
    PeftModelForSeq2SeqLM,                   # 面向 Seq2Seq LM 任务的包装类。
    PeftModelForSequenceClassification,      # 面向序列分类任务的包装类。
    PeftModelForTokenClassification,         # 面向标注任务（Token 分类）的包装类。
    PeftModelForCausalLMLORAMOE,             # 自定义/扩展：面向 Causal LM + LoRA-MoE(如 CoIN) 的包装类。
)
from .tuners import (                        # 从 tuners 子模块导入各种 PEFT 配置（超参/结构）类。
    AdaLoraConfig,                           # AdaLoRA 的配置。
    AdaptionPromptConfig,                    # Adaption Prompt 的配置。
    IA3Config,                               # IA3 的配置。
    LoraConfig,                              # LoRA 的配置。
    PrefixTuningConfig,                      # Prefix Tuning 的配置。
    PromptEncoderConfig,                     # P-Tuning v2（Prompt Encoder）的配置。
    PromptTuningConfig,                      # Prompt Tuning 的配置。
    CoINMOELoraConfig,                       # 自定义/扩展：CoIN-MoE LoRA 的配置。
)
from .utils import PromptLearningConfig, _prepare_prompt_learning_config
# PromptLearningConfig：所有 prompt 学习类方法（Prompt/Prefix/P-tuning）的共同基类配置。
# _prepare_prompt_learning_config：根据底模 config 对 Prompt 类配置做补全/校正的内部工具函数。

if TYPE_CHECKING:                            # 仅在类型检查阶段导入，运行时不执行，避免运行时依赖/开销。
    from transformers import PreTrainedModel # 只是给类型注解用：HuggingFace 的基类模型。
    from .utils.config import PeftConfig     # 同理，仅用于类型注解：PEFT 的通用配置基类。

# 将“任务类型字符串”映射到“具体的 PEFT 模型包装类”。任务类型通常来自 peft_config.task_type。
MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "SEQ_CLS": PeftModelForSequenceClassification,  # 序列分类 -> 对应包装类
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,         # Seq2Seq 语言模型 -> 包装类
    "CAUSAL_LM": PeftModelForCausalLM,             # 自回归语言模型 -> 包装类
    "TOKEN_CLS": PeftModelForTokenClassification,  # Token 分类 -> 包装类
    "QUESTION_ANS": PeftModelForQuestionAnswering, # 问答 -> 包装类
    "FEATURE_EXTRACTION": PeftModelForFeatureExtraction, # 特征提取 -> 包装类
    "CAUSAL_LM_CoIN": PeftModelForCausalLMLORAMOE, # 自定义扩展：Causal LM + CoIN/MoE LoRA -> 包装类
}

# 将“PEFT 方法类型字符串（peft_type）”映射到“对应的配置类”。用于把字典反序列化成 config 实例。
PEFT_TYPE_TO_CONFIG_MAPPING = {
    "ADAPTION_PROMPT": AdaptionPromptConfig,
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LORA": LoraConfig,
    "ADALORA": AdaLoraConfig,
    "IA3": IA3Config,
    "MOE_LORA_CoIN": CoINMOELoraConfig,    # 自定义扩展：MoE LoRA（CoIN）。
}

def get_peft_config(config_dict: Dict[str, Any]):
    """
    Returns a Peft config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """
    # 从传入字典里读取 "peft_type"，在上面的映射表里找到对应的配置类，并用 **config_dict 构造一个配置实例返回。
    # 注意：若缺少 "peft_type" 键或值不在映射表中，会抛异常（KeyError）。
    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)

def get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> PeftModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    # 取出底模的 config；若没有 config 属性，就用一个包含 {"model_type": "custom"} 的兜底字典。
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):      # 若 config 是 HF 的模型配置对象，转成 dict（便于后续处理）。
        model_config = model_config.to_dict()
    
    # 把底模的 name_or_path 写回到 peft_config 里，便于追踪基座模型来源（如保存/日志记录）。
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    # 分支1：如果 peft_config.task_type 不在映射表，且 peft_config 也不是 PromptLearningConfig（即非 Prompt 类方法）
    # 则返回通用 PeftModel（不做任务特化包装）。
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not isinstance(
        peft_config, PromptLearningConfig
    ):
        return PeftModel(model, peft_config, adapter_name=adapter_name)

    # 分支2：如果是 Prompt 类方法（Prompt/Prefix/P-Tuning 等），先根据底模 config 对其进行补齐/标准化。
    if isinstance(peft_config, PromptLearningConfig):
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)

    # 分支3：常规情况。根据 task_type 找到对应的“任务特化”包装类，实例化并返回。
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)
