"""
把 qwen_data/{train,test}/<task>.json (Qwen 格式 {problem, solution, image})
转成 LLaVA conversations 格式. 输出到同名目录的 ./llava_<split>/ 下。

LLaVA 期望:
{
  "id": "...",
  "image": "...",          # 可选: 纯文本任务无此字段
  "conversations": [
      {"from": "human", "value": "<image>\n<problem>"},
      {"from": "gpt",   "value": "<solution>"}
  ]
}
"""
import argparse, json, os


def convert(in_path, out_path):
    with open(in_path) as f:
        d = json.load(f)
    out = []
    for i, sample in enumerate(d):
        problem = sample.get("problem") or sample.get("original_question") or ""
        solution = sample.get("solution") or sample.get("original_answer") or ""
        item = {
            "id": sample.get("question_id", f"{os.path.basename(in_path)}_{i}"),
            "conversations": [],
        }
        image = sample.get("image")
        if image:
            item["image"] = image
            human = "<image>\n" + problem
        else:
            human = problem
        item["conversations"] = [
            {"from": "human", "value": human},
            {"from": "gpt",   "value": solution},
        ]
        out.append(item)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  {in_path} → {out_path}   ({len(out)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data")
    ap.add_argument("--tasks", default="numglue,art,math,fomc,medical,OCR,science")
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    for s in splits:
        for t in tasks:
            in_p = os.path.join(args.root, s, f"{t}.json")
            if not os.path.isfile(in_p):
                print(f"[skip] missing {in_p}")
                continue
            out_p = os.path.join(args.root, f"llava_{s}", f"{t}.json")
            convert(in_p, out_p)


if __name__ == "__main__":
    main()
