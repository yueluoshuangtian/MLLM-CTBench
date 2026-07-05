import os
import argparse
import json
import re
import string
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator, GQAAccuracyEvaluator

letters = string.ascii_uppercase


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--dataset', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    pattern = r"Question: (.*?) Options:"
    match = re.search(pattern, prompt, re.DOTALL)
    if match is not None:
        prompt = match.group(1)

    pattern = r"Question: (.*?) Short answer:"
    match = re.search(pattern, prompt, re.DOTALL)
    if match is not None:
        prompt = match.group(1)

    return prompt.lower()


def eval_single(annotation_file, result_file, dataset):
    annotations = json.load(open(annotation_file))
    if dataset == 'textvqa':
        annotations = annotations['data']
    if dataset == "textvqa":
        annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    elif "question_id" in annotations[0]:
        annotations = {(str(annotation['question_id']), annotation['question'].lower()): annotation for annotation in annotations}
    else:
        annotations = {(i, annotation['question'].lower()): annotation for i, annotation in enumerate(annotations)}

    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
        pred_answer = result['text']

        if dataset in ["vqav2", "okvqa", "ocrvqa"]:
            gt_answers = annotation['answer']
        elif dataset == "textvqa":
            gt_answers = annotation['answers']
        elif dataset == "gqa":
            gt_answers = [annotation['answer']]
        elif dataset == "vizwiz":
            gt_answers = [ans["answer"] for ans in annotation['answers']]
        elif dataset == "aokvqa":
            gt_answers = [letters[annotation["correct_choice_idx"]]]
            pred_choice = re.findall(r"(?<=\().*?(?=\))", pred_answer)
            pred_answer = pred_choice[0] if len(pred_choice) > 0 else pred_answer
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        pred_list.append({
            "pred_answer": pred_answer,
            "gt_answers": gt_answers,
        })

    if dataset in ["vqav2", "vizwiz", "textvqa", "okvqa"]:
        evaluator = TextVQAAccuracyEvaluator()
    elif dataset in ["gqa", "ocrvqa", "aokvqa"]:
        evaluator = GQAAccuracyEvaluator()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    accuracy = evaluator.eval_pred_list(pred_list)
    print('Samples: {}, Accuracy: {:.2f}%\n'.format(len(pred_list), 100. * accuracy))

    return accuracy * 100.


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file, args.dataset)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file), args.dataset)
