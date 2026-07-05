import json
from process_json_to_use import load_jsonl_datas
import os
import glob
from tqdm import tqdm
class find_wrond():
    def __init__(self,dirctft_results_filepath,CoT_results_filepath,CoT_json):
        self.dirctft_results_filepath = dirctft_results_filepath
        self.CoT_results_filepath = CoT_results_filepath
        self.cot_json = CoT_json


    def find_different_wrong_index(self):
        dirctft_result_datas = load_jsonl_datas(self.dirctft_results_filepath)
        cot_result_datas = load_jsonl_datas(self.CoT_results_filepath)
        dirctft_index = [data["question_id"] for data in dirctft_result_datas]
        cot_index = [data["question_id"] for data in cot_result_datas]
        different_indexs = [item for item in cot_index if item not in dirctft_index]
        index_labels = {}
        for data in cot_result_datas:
            if data['question_id'] in different_indexs:
                index_labels[data["question_id"]] = data['label']
        return different_indexs,index_labels

    def find_wrong_vqa(self):

        different_indexs,index_labels = self.find_different_wrong_index()
        cot_datas = load_jsonl_datas(self.cot_json)
        new_datas = [
            {**data, "label": index_labels[data['question_id']]}
            for data in cot_datas
            if data['question_id'] in different_indexs
                    ]
        subject = self.CoT_results_filepath.split("/")[-2].replace("test_","")
        save_path = os.path.join("/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/wrong_answer_analyze",f'{subject}')
        extension = os.path.split(self.cot_json)[1]
        os.makedirs(save_path,exist_ok=True)
        save_filename = os.path.join(save_path,extension)

        with open(save_filename,'w')as f:
            json.dump(new_datas,f,indent=4)
        return new_datas
    
if __name__ == "__main__":
    # subjects = ['test_fomc','test_ocr','test_art']
    ori_filepath = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool'
    for filepath in tqdm(glob.glob("/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_**",recursive=True)):
        if "test_object" in filepath:
            continue

        CoT_results_filepath = glob.glob(f"{filepath}/test_**reverse_results.jsonl",recursive=True)[0]
        dirctft_results_filepath = glob.glob(f"{filepath}/test_ft_eval_results.jsonl",recursive=True)[0]
        CoT_json = glob.glob(f"{filepath}/test_**reverse.jsonl",recursive=True)[0]
        analyze_wrong_answer = find_wrond(dirctft_results_filepath,CoT_results_filepath,CoT_json)
        new_datas = analyze_wrong_answer.find_wrong_vqa()
    