from evaluate import eval_results
if __name__ == '__main__':
    # use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/infer_answer_use.json'
    # eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ori_eval.jsonl'
    # save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_ori_results.jsonl'
    # eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)
    # use_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/samples/samples_no_CoT/OCR_samples_change.json'
    # eva_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/train_no_CoT.jsonl'
    # save_json_file_path = '/public/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/train_no_CoT_results.jsonl'
    # eval_results(use_json_file_path,eva_json_file_path,save_json_file_path,is_train_sample=True)
    use_json_file_path = '/home/houzhiyan/dataset/llava/infer_answers/ocr/test_infer_answer.json'
    eva_json_file_path = '/home/houzhiyan/HiDe_LLava_outputs/order2/model_outputs/OCR_adapt/0/merge.jsonl'
    save_json_file_path = '/home/houzhiyan/llava/llava_datasets/new_eval_tool/test_ocr/test_10epoch_revserse_results.jsonl'
    eval_results(use_json_file_path,eva_json_file_path,save_json_file_path)