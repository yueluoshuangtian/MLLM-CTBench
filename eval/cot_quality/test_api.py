"""
连通性自测：给 key 后先跑这个，确认 base_url / key / 三个模型名都可用。
用法：
    export NEWAPI_BASE_URL=...   export NEWAPI_KEY=...
    python test_api.py
"""
import config
from newapi_client import chat, list_models


def main():
    print("BASE_URL:", config.BASE_URL or "(未设置!)")
    print("KEY 是否设置:", bool(config.API_KEY))
    print("\n--- 拉取可用模型列表（前 40 个）---")
    try:
        models = list_models()
        for m in sorted(models)[:40]:
            print("  ", m)
        print(f"  ...共 {len(models)} 个")
    except Exception as e:
        print("  列模型失败（不影响调用，部分网关禁用 /models）:", e)

    print("\n--- 逐个测试 3 个 judge 模型 ---")
    for name, model in config.JUDGES.items():
        key = config.JUDGE_KEYS.get(name) or None
        try:
            text, usage = chat(model, [
                {"role": "user", "content": "Reply with exactly: OK"}
            ], api_key=key)
            print(f"  [{name}] {model} -> {text!r}  usage={usage}")
        except Exception as e:
            print(f"  [{name}] {model} -> 失败: {type(e).__name__}: {e}")
            print(f"        若是模型名错误，用上面列表里的真实名字设 NEWAPI_MODEL_{name.upper()}")


if __name__ == "__main__":
    main()
