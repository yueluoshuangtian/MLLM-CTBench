import json

from tqdm import tqdm
def load_jsonl_datas(jsonl_filepath):
    # 用于存储解析后的 JSON 对象
    datas = []
    
    # 逐行读取并解析 JSON Lines 文件
    with open(jsonl_filepath, 'r') as f:
        for line in f:
            # 去掉前后空白字符
            line = line.strip()
            if line:  # 确保行不为空
                try:
                    # 解析 JSON 对象并添加到列表中
                    data = json.loads(line)
                    datas.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(e)
    return datas
def read_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_datas = []
        # 逐行读取文件
        for line in file:
            # 将每行的内容解析为 JSON 对象
            data = json.loads(line)
            # 输出所需的信息
            question_id = data.get('question_id', 'N/A')
            image = data.get('image', 'N/A')
            text = data.get('text', 'N/A')
            answer = data.get('answer','N/A')
            new_dict = {'question_id':question_id,
                        'image':image,
                        'text':text,
                        'answer':answer}
            json_datas.append(new_dict)
    return json_datas

def process_use(answer_file_name,eva_question_name):
    # pdb.set_trace()
    json_datas = read_json_lines(answer_file_name)
    jsonl_datas = load_jsonl_datas(eva_question_name)
    #首先找到question和id的对应
    ids_questions_dict = {}
    ids_answers_dict = {}
    questions_ids = []
 
    for jsonl_data in jsonl_datas:
        questions_ids.append(jsonl_data['question_id'])
        
        ids_questions_dict[jsonl_data['question_id']] = jsonl_data["prompt"]
        ids_answers_dict[jsonl_data['question_id']] = jsonl_data['text']
        
    #通过上面给出id和question的对应
    ids_labels_dict = {}
    for json_data in json_datas:
        ids_labels_dict[json_data['question_id']] = json_data["answer"]
        

    return ids_labels_dict,ids_answers_dict,questions_ids,ids_questions_dict
                
if __name__ == "__main__":
    use_file_name = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_object/test_use.json'
    eva_name= '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_object/test_ori_eval.jsonl'
    a,b,c= process_use(use_file_name,eva_name)

