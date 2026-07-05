import json
import re
from tqdm import tqdm
from process_json import save_loads_json
from process_json_to_use import process_use
import logging
from rouge import Rouge 
from nltk.translate.bleu_score import corpus_bleu
import sys
sys.setrecursionlimit(100000) #例如这里设置为十万 
commaStrip= re.compile(r"(\d)(\,)(\d)")
periodStrip= re.compile(r"(?!<=\d)(\.)(?!\d)")
punct= [';', r"/", '[', ']', '"', '{', '}','(', ')', '=', '+', '\\', '_', '-','>', '<', '@', '`', ',', '?', '!']
manualMap= { 'none': '0','zero': '0','one': '1','two': '2','three': '3','four': '4','five': '5',
							  'six': '6','seven': '7','eight': '8','nine': '9','ten': '10'}
articles= ['a','an','the']
contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}
def processPunctuation( inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",
                                   outText,
                                   re.UNICODE)
    return outText

def evaluate_average_rouge_l(questions_ids,answers_index,real_answers_index):
     # 初始化 ROUGE 对象
    rouger = Rouge()

    # 初始化累加变量
    all_rouge1_f1 = 0
    all_rouge1_p = 0
    all_rouge1_r = 0

    all_rouge2_f1 = 0
    all_rouge2_p = 0
    all_rouge2_r = 0

    all_rougel_f1 = 0
    all_rougel_p = 0
    all_rougel_r = 0

    all_bleu_score = 0

    all_bleu_score = 0
    candidate_summaries = []
    reference_summaries = []
    for question_id in tqdm(questions_ids):      
        # 计算 ROUGE 分数
        rouge_scores = rouger.get_scores(answers_index[question_id], real_answers_index[question_id], avg=True)
        # 累加 ROUGE-1 分数
        all_rouge1_f1 += rouge_scores['rouge-1']['f']
        all_rouge1_p += rouge_scores['rouge-1']['p']
        all_rouge1_r += rouge_scores['rouge-1']['r']
        
        # 累加 ROUGE-2 分数
        all_rouge2_f1 += rouge_scores['rouge-2']['f']
        all_rouge2_p += rouge_scores['rouge-2']['p']
        all_rouge2_r += rouge_scores['rouge-2']['r']
        
        # 累加 ROUGE-L 分数
        all_rougel_f1 += rouge_scores['rouge-l']['f']
        all_rougel_p += rouge_scores['rouge-l']['p']
        all_rougel_r += rouge_scores['rouge-l']['r']
        candidate_summaries.append(answers_index[question_id].split())
        reference_summaries.append([real_answers_index[question_id].split()])


    # 计算平均 ROUGE 分数
    num_questions = len(questions_ids)
    average_bleu_score = corpus_bleu(reference_summaries, candidate_summaries) * 100
    average_rouge1_f1 = 100 * all_rouge1_f1 / num_questions
    average_rouge1_p = 100 * all_rouge1_p / num_questions
    average_rouge1_r = 100 * all_rouge1_r / num_questions

    average_rouge2_f1 = 100 * all_rouge2_f1 / num_questions
    average_rouge2_p = 100 * all_rouge2_p / num_questions
    average_rouge2_r = 100 * all_rouge2_r / num_questions

    average_rougel_f1 = 100 * all_rougel_f1 / num_questions
    average_rougel_p = 100 * all_rougel_p / num_questions
    average_rougel_r = 100 * all_rougel_r / num_questions

    # 计算平均 BLEU 分数
    average_bleu_score = corpus_bleu(reference_summaries, candidate_summaries) * 100

    # 打印结果
    # print(f"Average ROUGE-1 F1 Score: {average_rouge1_f1:.2f}")
    # print(f"Average ROUGE-1 Precision: {average_rouge1_p:.2f}")
    # print(f"Average ROUGE-1 Recall: {average_rouge1_r:.2f}")

    # print(f"Average ROUGE-2 F1 Score: {average_rouge2_f1:.2f}")
    # print(f"Average ROUGE-2 Precision: {average_rouge2_p:.2f}")
    # print(f"Average ROUGE-2 Recall: {average_rouge2_r:.2f}")

    print(f"Average ROUGE-L F1 Score: {average_rougel_f1:.2f}")
    # print(f"Average ROUGE-L Precision: {average_rougel_p:.2f}")
    # print(f"Average ROUGE-L Recall: {average_rougel_r:.2f}")

    # print(f"Average BLEU Score: {average_bleu_score:.2f}")
        # save_loads_json(wrong_QA,save_json_file_path)
def processDigitArticle( inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText
def open_json_file(file_path):
    with open(file_path,'r') as f:
        datas = json.load(f)
    return datas

def createIndex(json_data):
    index = {}
    for item in json_data:
        index[item["question_id"]] = item
    return index

def index_ande_questions_ids(eval_datas ,answers_data,real_answers_data):
    # 创建索引
    text_index = {eval_data['question_id']: eval_data['text'] for eval_data in eval_datas}
    answers_index = {answer_data['question_id']: answer_data['answer'] for answer_data in answers_data}
    real_answers_index = {real_answer_data['question_id']: real_answer_data['label'] for real_answer_data in real_answers_data}
    question_ids_answer = [data['question_id'] for data in answers_data]
    question_ids_label = [data['question_id'] for data in real_answers_data]
    if not set(question_ids_answer) == set(question_ids_label):
        raise ValueError("预测结果和标签中question_id不一致")
    return text_index,answers_index,real_answers_index,question_ids_answer
def process_reasoning(answers_index):
    new_dict = {}
    for key,value in answers_index.items():
        if not "answer:" in value:
            print(key)
            answer = "wrong"
        else:
            answer = value.split('answer:')[-1]

        new_dict[key] = answer
    return new_dict
def reverse_answer_CoT(answers_index):
    new_dict = {}
    for key,value in answers_index.items():
        if not "answer:" in value:
            print(key)
            answer = "wrong"
        else:
            answer = value.split('answer:')[-1]
            answer = answer.split('.')[0]
        new_dict[key] = answer
    return new_dict
#适用于只包含选择、yes、no等短回答的问题，其他长文本不适用
def eval_results(use_json_file_path, eva_json_file_path, save_json_file_path,
                 is_reasoning=False, is_train_sample=False,
                 evaluate_rouge_fl=False, reverse=False):

    def normalize_text(x):
        """
        将各种可能的输入（None、list、数值、空白字符串等）规范为安全的、可比较的字符串。
        规则：
          - None -> ""
          - list -> 取第一个元素再递归规范化
          - 其他非字符串 -> str(x)
          - 字符串 -> strip 后，如果为空则返回 ""
          - 最终小写
        """
        if x is None:
            return ""
        if isinstance(x, list):
            return normalize_text(x[0] if len(x) > 0 else "")
        if not isinstance(x, str):
            x = str(x)
        x = x.strip()
        if x == "":
            return ""
        return x.lower()

    real_answers_index, answers_index, questions_ids, text_index = process_use(
        use_json_file_path, eva_json_file_path
    )

    print("computing accuracy")

    if is_reasoning:
        answers_index = process_reasoning(answers_index)
    if is_train_sample:
        real_answers_index = process_reasoning(real_answers_index)
    if reverse:
        answers_index = reverse_answer_CoT(answers_index)

    if evaluate_rouge_fl:
        evaluate_average_rouge_l(questions_ids, answers_index, real_answers_index)
        # 你原来 evaluate_average_rouge_l 的分支没有返回 acc；保持原样
        return None
    else:
        # 清理 & 规范化
        for question_id in questions_ids:
            # —— 预测答案：只取第一个 token；若为空则得到 ""（不再报错）
            pred_raw = answers_index.get(question_id, "")
            pred_norm = normalize_text(pred_raw)
            if pred_norm == "":
                # 保持与原来的“取第一个 token”的语义一致——空就仍为空
                answers_index[question_id] = ""
            else:
                # 只取第一个 token
                answers_index[question_id] = pred_norm.split()[0]

            # —— 真实答案：同样稳健规范化（兼容 list/None）
            gt_raw = real_answers_index.get(question_id, "")
            gt_norm = normalize_text(gt_raw)

            # y/n 归一化（修正了原来的 == 误用）
            if gt_norm == "y":
                gt_norm = "yes"
            elif gt_norm == "n":
                gt_norm = "no"

            real_answers_index[question_id] = gt_norm

        # 统计
        wrong_QA = [
            {
                "question_id": qid,
                "label": real_answers_index[qid],
                "answer": answers_index[qid],
                "prompt": text_index[qid],
            }
            for qid in tqdm(questions_ids)
            if real_answers_index[qid] != answers_index[qid]
        ]

        rightQA = [
            {
                "question_id": qid,
                "label": real_answers_index[qid],
                "answer": answers_index[qid],
            }
            for qid in tqdm(questions_ids)
            if real_answers_index[qid] == answers_index[qid]
        ]

        n = 2
        acc = round(100 * float(len(rightQA)) / max(1, len(questions_ids)), n)
        print(acc, "%")
        save_loads_json(wrong_QA, save_json_file_path)
        return acc

logging.basicConfig(level=logging.INFO,                   # 设置日志记录级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志记录格式
    filename='example.log',               # 日志文件名
    filemode='a' )

if __name__ == '__main__':
 
    use_json_file_path = '/home/houzhiyan/dataset/llava/infer_answers/math/test_infer_answer.json'
    eva_json_file_path = '/home/houzhiyan/HiDe_LLava_outputs/order2/model_outputs/math_adapt/0/merge.jsonl'
    save_json_file_path = '/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_math/test_reasoning_reaverse_eval_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)
