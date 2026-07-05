from evaluate import eval_results
if __name__ == '__main__':
    # use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_numberglum/NumGLUE_test_answer.json'
    # eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_numberglum/test_ori_eval.jsonl'
    # save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_numberglum/test_ori_eval_results.jsonl'
    # eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)
    # use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/samples/samples_no_CoT/numglue_samples_change.json'
    # eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_numberglum/train_no_CoT.jsonl'
    # save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_numberglum/train_no_CoT_result.jsonl'
    # eval_results(use_json_file_path,eva_json_file_path,save_json_file_path,is_train_sample=True)
    use_json_file_path = "/home/houzhiyan/dataset/llava/infer_answers/numglue/NumGLUE_test_answer.json"
    eva_json_file_path = '/home/houzhiyan/HiDe_LLava_outputs/order2/model_outputs/Numglue_adapt/0/merge.jsonl'
    save_json_file_path = '/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_numberglum/test_reasoning_reaverse_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)