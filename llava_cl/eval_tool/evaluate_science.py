
from evaluate_math import eval_results
if __name__ == '__main__':

    use_json_file_path = '/home/houzhiyan/dataset/llava/infer_answers/science/test_infer_answer.json'
    eva_json_file_path = '/home/houzhiyan/HiDe_LLava_outputs/order2/model_outputs/science_adapt/0/merge.jsonl'
    save_json_file_path = '/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_science/test_reverse_science_reasoning_10epoch_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)