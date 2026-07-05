import json
import re
from tqdm import tqdm
from process_json import save_loads_json
from process_json_to_use import process_use
from evaluate_art_long_sentences import evaluate_average_rouge_l
from evaluate_math import reverse_answer_CoT
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

def eval_results(use_json_file_path,eva_json_file_path,save_json_file_path,is_reasoning = False,is_train_sample=False,
                 evaluate_rouge_fl = False,reverse=False):

    real_answers_index,answers_index,questions_ids,text_index = process_use(use_json_file_path,eva_json_file_path)

    print("样本数量:", len(questions_ids))
    print("real_answers_index 示例:")
    for i, qid in enumerate(questions_ids[:3]):
        print(qid, "=>", real_answers_index[qid])
    print("answers_index 示例:")
    for i, qid in enumerate(questions_ids[:3]):
        print(qid, "=>", answers_index[qid])
    real_answers_index,answers_index,questions_ids,text_index = process_use(use_json_file_path,eva_json_file_path)
    # =================================================
    # Compute accuracy
    # =================================================
    accQA = []
    rightQA = []
    wrong_QA = []
    accAnsType = {}
    print("computing accuracy")
    # print("real_answers_index:",real_answers_index)
    # print("answers_index:",answers_index)
    step = 0
    if is_reasoning:
        answers_index = process_reasoning(answers_index)
    if is_train_sample:
        real_answers_index = process_reasoning(real_answers_index)
    if reverse:
        answers_index = reverse_answer_CoT(answers_index)
    if evaluate_rouge_fl:
        evaluate_average_rouge_l(questions_ids,answers_index,real_answers_index)
    else:
        #清理
        for question_id in questions_ids:
            answers_index[question_id] = answers_index[question_id].replace('\n', ' ')
            answers_index[question_id] = answers_index[question_id].replace('\t', ' ')
            answers_index[question_id] = answers_index[question_id].strip()
            answers_index[question_id] = processPunctuation(answers_index[question_id])
            if not len(processDigitArticle(answers_index[question_id])) == 0:
                answers_index[question_id] = processDigitArticle(answers_index[question_id])
            if isinstance(real_answers_index[question_id],list):
                real_answers_index[question_id] = real_answers_index[question_id][0]
            real_answers_index[question_id] = real_answers_index[question_id].replace('\n', ' ')
            real_answers_index[question_id] = real_answers_index[question_id].replace('\t', ' ')
            real_answers_index[question_id] = real_answers_index[question_id].strip()
            real_answers_index[question_id] = processPunctuation(real_answers_index[question_id])
            if not len(processDigitArticle(real_answers_index[question_id])) == 0:
                real_answers_index[question_id] = processDigitArticle(real_answers_index[question_id])
        wrong_QA = [{'question_id':question_id,'label':real_answers_index[question_id],'answer':answers_index[question_id],
                    'prompt':text_index[question_id]} for question_id in tqdm(questions_ids)
                    if real_answers_index[question_id]!=answers_index[question_id]]
        rightQA = [{'question_id':question_id,'label':real_answers_index[question_id],'answer':answers_index[question_id]} for question_id in tqdm(questions_ids)
                    if real_answers_index[question_id]==answers_index[question_id]]

        n = 2
        acc = round(100*float(len(rightQA))/len(questions_ids),n)
        print(acc,'%')
        save_loads_json(wrong_QA,save_json_file_path)
        return acc
if __name__ == '__main__':
    # use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_object/test_use.json'
    # eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_object/test_ori_eval.jsonl'
    # save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_object/test_ori_eval_results.jsonl'
    # eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)
    use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_object/test_use.json'
    eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_object/test_ft_eval.jsonl'
    save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_object/test_ft_eval_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path,reverse=True)


