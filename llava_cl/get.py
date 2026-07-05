import os
import json
import random
import glob
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

def get(filepath_1,filepath_2):
    datas_1 = load_jsonl_datas(filepath_1)
    datas_2 = load_jsonl_datas(filepath_2)
    question_ids = [data['question_id'] for data in datas_1]
    random.shuffle(question_ids)
    half_length = len(question_ids) // 3
    pre_question_ids = question_ids[:half_length]
    last_question_ids = question_ids[half_length:]
    
    pre_datas = [data for data in datas_1 if data['question_id'] in pre_question_ids]
    last_datas = [data for data in datas_2 if data['question_id'] in last_question_ids]
    
    all_datas = pre_datas + last_datas
    
    with open(filepath_1,'w') as f:
        for item in all_datas:
            json_str = json.dumps(item)
            f.write(json_str + "\n")
            
def main(filepath_1,filepath_2,task):
  
    tasks = ['fomc']
    last_tasks = ['fomc']
    
    if task == 'adapt':
        use_tasks = tasks
        filepath_1 = os.path.join(filepath_1,'predictions')
        filepath_2 = os.path.join(filepath_2,'predictions')
    elif task == 'last':
        use_tasks = last_tasks
        filepath_1 = os.path.join(filepath_1,'predictions_test')
        filepath_2 = os.path.join(filepath_2,'predictions_test')
    for use_task in use_tasks:
        filename_1 = glob.glob(f"{filepath_1}/*{use_task}*",recursive=True)[0]
        filename_2 = glob.glob(f"{filepath_2}/*{use_task}*",recursive=True)[0]
        get(filename_1,filename_2)
        
def seq_main(filepath_1,filepath_2):
    
    main(filepath_1,filepath_2,'adapt')
    main(filepath_1,filepath_2,'last')
    
if __name__ == "__main__":
    filepath_1 = "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/checkpoints/llava-v1.5-7b/order2/replay"
    filepath_2 = "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/checkpoints/llava-v1.5-7b/order2/replay_1e-2"
    main(filepath_1,filepath_2,'last')
    