"""
new-api 统一客户端：三个 judge（GPT/Claude/Gemini）都走 OpenAI 兼容端点
{BASE_URL}/v1/chat/completions，只切换 model 名。鉴权 Authorization: Bearer {KEY}。
"""
import time
import base64
import os
from openai import OpenAI

import config

_clients = {}  # api_key -> OpenAI client（支持每个 judge 用不同令牌）


def get_client(api_key=None):
    """惰性创建 OpenAI SDK 客户端，base_url 指向 new-api 网关。
    api_key 为空时回退到 config.API_KEY（统一令牌）。"""
    key = api_key or config.API_KEY
    if not config.BASE_URL or not key:
        raise RuntimeError(
            "请先设置环境变量 NEWAPI_BASE_URL 和 NEWAPI_KEY（见 config.py 注释）"
        )
    if key not in _clients:
        _clients[key] = OpenAI(
            api_key=key,
            base_url=config.BASE_URL + "/v1",
            timeout=config.REQUEST_TIMEOUT,
        )
    return _clients[key]


def list_models(api_key=None):
    """列出网关上当前可用的模型名，方便确认 JUDGES 里的模型名是否正确。"""
    client = get_client(api_key)
    return [m.id for m in client.models.list().data]


def encode_image(image_path):
    """把本地图片编码成 OpenAI 多模态所需的 data URI；失败/过大返回 None。"""
    if not image_path or not os.path.isfile(image_path):
        return None
    if os.path.getsize(image_path) > config.MAX_IMAGE_BYTES:
        return None
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def build_content(text, image_data_uri=None):
    """构造单条 user 消息的 content：纯文本或 文本+图片。"""
    if image_data_uri is None:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ]


def chat(model, messages, temperature=None, max_retries=None, api_key=None):
    """带重试的非流式 chat completion。返回 (text, usage_dict) 或抛出最后一次异常。"""
    client = get_client(api_key)
    temperature = config.TEMPERATURE if temperature is None else temperature
    max_retries = config.MAX_RETRIES if max_retries is None else max_retries
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            text = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            usage_dict = (
                {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens}
                if usage else {}
            )
            return text, usage_dict
        except Exception as e:  # noqa: BLE001 网关错误五花八门，统一退避重试
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    raise last_err
