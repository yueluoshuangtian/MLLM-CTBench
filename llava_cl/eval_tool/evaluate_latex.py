import json
import re
from tqdm import tqdm
from process_json import save_loads_json
from process_json_to_use import process_use
from difflib import SequenceMatcher
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
def calculate_err(expr1, expr2):
    """
    Calculate the Expression Recognition Rate (ERR) between two LaTeX expressions.
    """
    matcher = SequenceMatcher(None, expr1, expr2)
    match = matcher.find_longest_match(0, len(expr1), 0, len(expr2))
    err = match.size / max(len(expr1), len(expr2))
    return err
def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance
def eval_results(use_json_file_path,eva_json_file_path,save_json_file_path):

    
    real_answers_index,answers_index,questions_ids,text_index = process_use(use_json_file_path,eva_json_file_path)
    save_questions_ids = []
    for question_id in questions_ids:
        if question_id.startswith('OCR_vqa_datasets/Test/CROHME'):
            save_questions_ids.append(question_id)
    
    # =================================================
    # Compute accuracy
    # =================================================
    accQA = []
    rightQA = []
    wrong_QA = []
    accAnsType = {}
    print("computing accuracy")
    step = 0
    all_acc = 0
    wrong_QA = []
    e1 = 0
    e2 = 0
    e3 = 0
    #清理
    for question_id in save_questions_ids:
        if answers_index[question_id] == real_answers_index[question_id]:
            all_acc += 1
        else:
            wrong_dict = {'question_id':question_id,'label':real_answers_index[question_id],'answer':answers_index[question_id],
                 'prompt':text_index[question_id]}
            wrong_QA.append(wrong_dict)
        # acc_single = calculate_err(answers_index[question_id],real_answers_index[question_id])
        # all_acc += acc_single
        distance = compute_edit_distance(answers_index[question_id], real_answers_index[question_id])
        if distance <= 1:
            e1 += 1
        if distance <= 2:
            e2 += 1
        if distance <= 3:
            e3 += 1

    n = 2
    ExpRate = round(100*float(all_acc)/len(save_questions_ids),n)
    ExpRate_1 = round(100*float(e1)/len(save_questions_ids),n)
    ExpRate_2 = round(100*float(e2)/len(save_questions_ids),n)
    ExpRate_3 = round(100*float(e3)/len(save_questions_ids),n)
    print('the ExpRate:',ExpRate,'%')
    print('<=1 error:',ExpRate_1,'%')
    print('<=2 error:',ExpRate_2,'%')
    print('<=3 error:',ExpRate_3,'%')

    save_loads_json(wrong_QA,save_json_file_path)

if __name__ == '__main__':
    use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/infer_answer_use.json'
    eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ori_eval.jsonl'
    save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ori_latex_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)
    
    use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/infer_answer_use.json'
    eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ft_eval.jsonl'
    save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ft_latex_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)

    use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/infer_answer_use.json'
    eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ft_all_eval.jsonl'
    save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ft_all_latex_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)

    use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/infer_answer_use.json'
    eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ft_6epoches_eval.jsonl'
    save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ft_6epoches_latex_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)
    