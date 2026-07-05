
from evaluate import eval_results
if __name__ == '__main__':
    
    use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_art/test_infer_answer.json'
    eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_art/test_reasoning_reaverse.jsonl'
    save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_art/test_reasoning_reaverse_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path,reverse=True)
    # use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_art/test_infer_answer.json'
    # eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_art/test_ft_reasoning_14epoches_eval.jsonl'
    # save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_art/test_ft_reasoning_14epoches_eval_results.jsonl'
    # eval_results(use_json_file_path,eva_json_file_path,save_json_file_path,is_reasoning=True)