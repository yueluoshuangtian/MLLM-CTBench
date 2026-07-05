
from evaluate import eval_results
if __name__ == '__main__':
    use_json_file_path = '/home/houzhiyan/dataset/llava/infer_answers/medical/test_infer_answer.json'
    eva_json_file_path = '/home/houzhiyan/fixed_experts_output/without_reasoning/llava-v1.5-7b/order1/moelora/answer/medical.jsonl'
    save_json_file_path = '/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_medical/test_reverse_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)