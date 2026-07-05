import os
import argparse
import json
import re
import string
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator, GQAAccuracyEvaluator, SciQAAccuracyEvaluator

letters = string.ascii_uppercase


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--dataset', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif "\nContext:" in prompt:    # scienceqa
        question = prompt.split("\nAnswer with the option's letter from the given choices directly.")[0]
    else:
        question = prompt.split('\n')[0]

    return question.lower()


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
        elif dataset == "scienceqa":
            pattern = re.compile(r'The answer is ([A-Z]).')
            gt_answers = pattern.findall(annotation['answer'])[0]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": gt_answers,
        })

    if dataset in ["vqav2", "vizwiz", "textvqa", "okvqa"]:
        evaluator = TextVQAAccuracyEvaluator()
    elif dataset in ["gqa", "ocrvqa", "aokvqa"]:
        evaluator = GQAAccuracyEvaluator()
    elif dataset == "scienceqa":
        evaluator = SciQAAccuracyEvaluator()
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
