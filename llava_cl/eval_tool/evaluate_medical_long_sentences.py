import json
import re
from tqdm import tqdm
from process_json import save_loads_json
from process_json_to_use import process_use
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge 
from evaluate_math import process_reasoning, reverse_answer_CoT

# 正则和标点定义
commaStrip = re.compile("(\d)(\,)(\d)")
periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
punct = [';', r"/", '[', ']', '"', '{', '}','(', ')', '=', '+', '\\', '_', '-','>', '<', '@', '`', ',', '?', '!']
manualMap = {'none': '0','zero': '0','one': '1','two': '2','three': '3','four': '4','five': '5',
             'six': '6','seven': '7','eight': '8','nine': '9','ten': '10'}
articles = ['a','an','the']
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

def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText

def processDigitArticle(inText):
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

# ==========================================================
#  修改后的核心函数：增加了空值检查和 try-except
# ==========================================================
def evaluate_average_rouge_l(questions_ids, answers_index, real_answers_index):
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

    candidate_summaries = []
    reference_summaries = []
    
    # 用于记录有效计算的样本数
    valid_count = 0 

    for question_id in tqdm(questions_ids):
        # 1. 安全获取文本，并处理 None
        pred_text = answers_index.get(question_id, "")
        ref_text = real_answers_index.get(question_id, "")
        
        if pred_text is None: pred_text = ""
        if ref_text is None: ref_text = ""

        # 2. 强制转字符串并清理首尾
        pred_text = str(pred_text).strip()
        ref_text = str(ref_text).strip()

        # 3. 检查是否为空
        # 如果预测为空，ROUGE 无法计算，按 0 分处理
        if not pred_text:
            # 预测为空，添加空列表以保持 BLEU 计算对齐
            candidate_summaries.append([])
            reference_summaries.append([ref_text.split() if ref_text else []])
            valid_count += 1
            continue
        
        # 如果标签为空，跳过该样本（或者也算0分，视需求而定，这里选择跳过不计入分母）
        if not ref_text:
            continue

        try:
            # 4. 计算 ROUGE 分数
            rouge_scores = rouger.get_scores(pred_text, ref_text, avg=True)

            # 累加 ROUGE-1
            all_rouge1_f1 += rouge_scores['rouge-1']['f']
            all_rouge1_p += rouge_scores['rouge-1']['p']
            all_rouge1_r += rouge_scores['rouge-1']['r']
            
            # 累加 ROUGE-2
            all_rouge2_f1 += rouge_scores['rouge-2']['f']
            all_rouge2_p += rouge_scores['rouge-2']['p']
            all_rouge2_r += rouge_scores['rouge-2']['r']
            
            # 累加 ROUGE-L
            all_rougel_f1 += rouge_scores['rouge-l']['f']
            all_rougel_p += rouge_scores['rouge-l']['p']
            all_rougel_r += rouge_scores['rouge-l']['r']

            candidate_summaries.append(pred_text.split())
            reference_summaries.append([ref_text.split()])
            
            valid_count += 1

        except Exception as e:
            # 捕获其他可能的错误（如分词后为空），视为 0 分
            # print(f"Error on ID {question_id}: {e}")
            candidate_summaries.append([])
            reference_summaries.append([ref_text.split()])
            valid_count += 1

    # 5. 计算平均值
    # 注意：使用 valid_count 作为分母
    if valid_count == 0:
        print("Warning: No valid samples found.")
        return 0.0

    average_rouge1_f1 = 100 * all_rouge1_f1 / valid_count
    # average_rouge1_p = 100 * all_rouge1_p / valid_count
    # average_rouge1_r = 100 * all_rouge1_r / valid_count

    # average_rouge2_f1 = 100 * all_rouge2_f1 / valid_count
    # average_rouge2_p = 100 * all_rouge2_p / valid_count
    # average_rouge2_r = 100 * all_rouge2_r / valid_count

    average_rougel_f1 = 100 * all_rougel_f1 / valid_count
    # average_rougel_p = 100 * all_rougel_p / valid_count
    # average_rougel_r = 100 * all_rougel_r / valid_count

    # 计算平均 BLEU 分数
    try:
        average_bleu_score = corpus_bleu(reference_summaries, candidate_summaries) * 100
    except:
        average_bleu_score = 0

    print(f"Processed {valid_count} samples.")
    print(f"Average ROUGE-L F1 Score: {average_rougel_f1:.2f}")
    print(f"Average BLEU Score: {average_bleu_score:.2f}")

    return round(average_rougel_f1, 2)

def eval_results(use_json_file_path, eva_json_file_path, save_json_file_path,
                 is_reasoning=False, is_train_sample=False,
                 evaluate_rouge_fl=False, reverse=False):
    
    real_answers_index, answers_index, questions_ids, text_index = process_use(use_json_file_path, eva_json_file_path)
    
    print("computing accuracy")
    
    if is_reasoning:
        answers_index = process_reasoning(answers_index)
    if is_train_sample:
        real_answers_index = process_reasoning(real_answers_index)
    if reverse:
        answers_index = reverse_answer_CoT(answers_index)
        
    if evaluate_rouge_fl:
        average_rouge_l = evaluate_average_rouge_l(questions_ids, answers_index, real_answers_index)
        return average_rouge_l
    else:
        # 清理逻辑
        for question_id in tqdm(questions_ids):
            # 防御性编程：处理 key 可能不存在或为 None 的情况
            pred_text = answers_index.get(question_id, "")
            if pred_text is None: pred_text = ""
            
            if len(pred_text) == 0:
                answers_index[question_id] = 'wrong answer'
                
            # 重新获取以确保是字符串
            temp_ans = str(answers_index[question_id])
            temp_ans = temp_ans.replace('\n', ' ').replace('\t', ' ').strip()
            temp_ans = processPunctuation(temp_ans)
            
            processed_digit = processDigitArticle(temp_ans)
            if len(processed_digit) > 0:
                answers_index[question_id] = processed_digit
            else:
                answers_index[question_id] = temp_ans

            # 处理标签
            real_val = real_answers_index.get(question_id, "")
            if isinstance(real_val, list):
                real_val = real_val[0] if len(real_val) > 0 else ""
            
            real_val = str(real_val).replace('\n', ' ').replace('\t', ' ').strip()
            real_val = processPunctuation(real_val)
            
            processed_digit_real = processDigitArticle(real_val)
            if len(processed_digit_real) > 0:
                real_answers_index[question_id] = processed_digit_real
            else:
                real_answers_index[question_id] = real_val
                
        average_rouge_l = evaluate_average_rouge_l(questions_ids, answers_index, real_answers_index)
        return average_rouge_l

if __name__ == '__main__':
    use_json_file_path = '/home/houzhiyan/dataset/llava/infer_answers/medical/test_infer_answer.json'
    eva_json_file_path = '/home/houzhiyan/HiDe_LLava_outputs/order2/model_outputs/medical_adapt/0/merge.jsonl'
    save_json_file_path = '/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_medical/test_reverse_results.jsonl'
    
    # 运行评估
    eval_results(use_json_file_path, eva_json_file_path, save_json_file_path, evaluate_rouge_fl=True)