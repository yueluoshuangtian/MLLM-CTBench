"""
7 个任务的 pointwise（单点）质量打分 rubric。

模仿 a_data_use/generate_{task}_reasoning.py 里 pairwise「contrast example」路径的
逐任务评分维度，但改为对**单条训练 CoT**打分（不再对比 response_1/response_2）。
已修正源码里的两个 bug：
  (1) "1." 被全局替换成 "Multi-digit decimal places."
  (2) Probability/QuantityNumber 的 system prompt 误写成 "Geometric Content in the Diagram"

维度统一映射到论文评测器三维度：
  logic     = 逻辑连贯 (所有任务)
  grounding = 视觉接地 (仅含图 VQA 任务；纯文本任务为 None)
  knowledge = 领域知识准确 (所有任务)
judge 必须输出严格 JSON，便于统计解析。
"""

# ---- 每个任务的三维度描述（pointwise 措辞，逐任务领域名不同）----
# 结构: task -> {"logic":..., "grounding":... or None, "knowledge":...}
# grounding=None 表示纯文本任务，省略该维度（与论文一致）。

TASK_DIMENSIONS = {
    "math_qa": {
        "logic": "Logical Coherence and Reasoning Flow: is the reasoning clear, coherent, and following a valid logical sequence to the conclusion?",
        "grounding": None,
        "knowledge": "Accuracy of Mathematical Concepts: are the calculations and mathematical concepts correct and properly applied?",
    },
    "economics_qa": {
        "logic": "Logical Coherence and Reasoning Flow: is the reasoning clear, coherent, and following a valid logical sequence to the conclusion?",
        "grounding": None,
        "knowledge": "Understanding of Economic Terminology: does the reasoning correctly understand and apply economic concepts (e.g., monetary policy stance)?",
    },
    "science_vqa": {
        "logic": "Logical Coherence and Reasoning Flow: is the reasoning clear, coherent, and logically valid?",
        "grounding": "Correct Identification of Elements in the Diagram: does the reasoning accurately identify and understand the elements shown in the image/diagram?",
        "knowledge": "Accuracy in the Application of {sub} Knowledge: is the domain (scientific) knowledge correctly defined and appropriately applied?",
    },
    "math_vqa": {
        "logic": "Logical Clarity and Completeness: how clearly and logically does the reasoning explain each step?",
        "grounding": "Correct Identification and Application of Visual Content: does the reasoning accurately interpret the chart/geometry/diagram elements in the image?",
        "knowledge": "Mathematical Accuracy and Solution Strategy: are the calculations, methods, and formulas correct?",
    },
    "medicine_vqa": {
        "logic": "Logical Coherence and Reasoning Flow: is the reasoning clear, coherent, and logically valid?",
        "grounding": "Interpretation and Application of Radiological Knowledge: does it correctly interpret the medical image (X-ray/CT/MRI/ultrasound) and use it appropriately?",
        "knowledge": "Accuracy of Medical Knowledge: are the anatomical, pathological, and radiological concepts applied accurately?",
    },
    "ocr_vqa": {
        "logic": "Logical Coherence and Reasoning Flow: is the reasoning clear, coherent, and logically valid?",
        "grounding": "Correct Identification of Visual Content: {sub}",
        "knowledge": "Correct Text Extraction: does it correctly extract and interpret the textual information from the image?",
    },
    "arts_vqa": {
        "logic": "Logical Coherence and Reasoning Flow: how well does the reasoning maintain clarity and a logical structure?",
        "grounding": "Image Interpretation and Artistic Analysis: how well does it interpret the artwork and analyze its artistic elements?",
        "knowledge": "Cultural and Contextual Insight: how well does it incorporate relevant cultural or historical context?",
    },
}

# ---- 子类检测（从 image 路径/question_id 推断），用于替换 {sub} 占位 ----
SCIENCE_SUBJECTS = ["geography", "chemistry", "biology", "engineering", "astronomy", "physics"]
MATH_TYPES = ["GeometryNumericCal", "PlotCompare", "BarPieCompare",
              "GeometryPositionShape", "Probability", "QuantityNumber"]
OCR_GROUNDING = {
    "ChartOCR": "does it accurately identify and understand the chart/diagram elements?",
    "CROHME":   "does it accurately identify the handwritten formula structure and convert it to correct LaTeX?",
    "scene":    "does it accurately identify complex scene elements (people, objects, environment) and their relationships?",
}


def detect_subtype(task, image_path, question_id=""):
    """根据路径推断子类，返回用于填充 rubric 的字符串描述。"""
    p = (image_path or "") + " " + (question_id or "")
    pl = p.lower()
    if task == "science_vqa":
        for s in SCIENCE_SUBJECTS:
            if s in pl:
                return s
        return "scientific"
    if task == "math_vqa":
        for t in MATH_TYPES:
            if t.lower() in pl:
                return t
        return "mathematical"
    if task == "ocr_vqa":
        if "chartocr" in pl:
            return OCR_GROUNDING["ChartOCR"]
        if "crohme" in pl:
            return OCR_GROUNDING["CROHME"]
        return OCR_GROUNDING["scene"]
    return ""


SYSTEM_PROMPT = (
    "You are a rigorous evaluator of Chain-of-Thought (CoT) reasoning quality. "
    "You will be given a question (optionally with an image) and a CoT reasoning "
    "annotation that was generated by GPT-4 and used as training data. Your job is to "
    "judge the QUALITY of that reasoning annotation along the given criteria. "
    "Be objective and calibrated. Output ONLY a JSON object, no extra text."
)


def build_messages(task, question, cot_text, image_data_uri=None,
                   ref_answer=None, image_path="", question_id=""):
    """构造一次 pointwise 打分的 messages（system + user）。返回 messages 列表。"""
    dims = TASK_DIMENSIONS[task]
    sub = detect_subtype(task, image_path, question_id)
    logic = dims["logic"]
    knowledge = dims["knowledge"]
    grounding = dims["grounding"]
    if grounding and "{sub}" in grounding:
        grounding = grounding.format(sub=sub)
    if "{sub}" in knowledge:
        knowledge = knowledge.format(sub=sub)

    # 组装维度清单 + 期望 JSON 字段
    crit_lines = [f"- logic: {logic}", f"- knowledge: {knowledge}"]
    if grounding:
        crit_lines.insert(1, f"- grounding: {grounding}")
    crit_block = "\n".join(crit_lines)

    json_fields = '"logic": <0-100>, '
    if grounding:
        json_fields += '"grounding": <0-100>, '
    json_fields += ('"knowledge": <0-100>, "overall": <0-100>, '
                    '"answer_consistent": <true|false>, "rationale": "<one sentence>"')

    ref_line = f"\nThe annotation concludes with this answer: {ref_answer}" if ref_answer else ""

    user_text = f"""### Task: judge the quality of the CoT reasoning annotation below.

Score each criterion from 0 to 100 (0-25 Irrelevant, 26-50 Partially Correct, 51-75 Mostly Correct, 76-100 Fully Correct):
{crit_block}

Also report:
- overall: an overall quality score 0-100.
- answer_consistent: whether the reasoning is internally consistent and correctly leads to its stated final answer (true/false).

### Question:
{question}
{ref_line}

### CoT reasoning annotation to evaluate:
{cot_text}

### Output (STRICT JSON only):
{{{json_fields}}}"""

    from newapi_client import build_content
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_content(user_text, image_data_uri)},
    ], bool(grounding)
