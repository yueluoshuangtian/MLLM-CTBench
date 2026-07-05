import json
import os
#按照要求保存文件
def save_loads_json(list,save_path):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    with open(save_path, "w") as f:
        for item in list:
            json.dump(item, f)
            f.write("\n")


def remove_image_extension(filename):
    """
    去掉图片文件名中最后的图片格式（扩展名）。

    参数:
    filename (str): 图片文件名。

    返回:
    str: 去掉扩展名后的文件名。
    """
    # 使用os.path.splitext方法分离文件名和扩展名
    name, ext = os.path.splitext(filename)

    # 返回不包含扩展名的文件名
    return name
#图片名称相同，其question_id应该不同
def change_question_id(leap_use_data):
    queston_ids = []
    id_count = {}
    noleap_json = []
    for data in leap_use_data:
        if data['question_id'] not in queston_ids:
            ori_question_id = data['question_id']
            queston_ids.append(ori_question_id)
            id_count[ori_question_id] = 0
            data['question_id'] = f'{ori_question_id}_0'
            noleap_json.append(data)
        else:
            ori_id_question = data['question_id']
            id_count[ori_id_question] += 1
            id_index = id_count[ori_id_question]
            data['question_id'] = f'{ori_id_question}_{id_index}'
            noleap_json.append(data)
    return noleap_json
def process_json_to_answer(file_path):
    with open(file_path,'r') as f:
        datas = json.load(f)
    save_jsons = []
    for data in datas:
        for conversion in data['conversations']:
            if conversion['from'] == 'gpt':
                dict ={
                    'question_id' : remove_image_extension(data['image']),
                    'image' : data['image'],
                    'label' : conversion['value']
                }
                save_jsons.append(dict)
    save_path = file_path.replace('.json','_label.json')
    save_jsons = change_question_id(save_jsons)
    with open(save_path,'w') as f:
        json.dump(save_jsons,f)
def process_json_to_text(file_path):
    with open(file_path,'r') as f:
        datas = json.load(f)
    save_jsons = []
    for data in datas:
        for conversion in data['conversations']:
            if conversion['from'] == 'human':
                dict ={
                    'question_id' : remove_image_extension(data['image']),
                    'image' : data['image'],
                    'text' : conversion['value'].replace("<image>\n", "")
                }
                save_jsons.append(dict)
    save_path = file_path.replace('.json','_text.json')
    save_jsons = change_question_id(save_jsons)
    with open(save_path,'w') as f:
        json.dump(save_jsons,f)
if __name__ == "__main__":
    process_json_to_answer('TestChartOCR.json')
    process_json_to_text('TestChartOCR.json')