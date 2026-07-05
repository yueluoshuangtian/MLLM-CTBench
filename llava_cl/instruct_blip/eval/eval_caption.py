import os
import argparse
import json
import re
import string
from llava.eval.m4c_evaluator import TextCapsCiderEvaluator

letters = string.ascii_uppercase


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--dataset', type=str)
    return parser.parse_args()


def eval_single(annotation_file, result_file, dataset):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))
    if dataset == 'textvqa':
        annotations = annotations['data']
    annotations = {i: annotation for i, annotation in enumerate(annotations)}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[result['question_id']]

        if dataset in ["textcaps", "flickr30k"]:
            gt_answers = annotation['caption']
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": gt_answers,
        })

    evaluator = TextCapsCiderEvaluator()
    score = evaluator.eval_pred_list(pred_list)
    print('CIDEr: {:.2f}\n'.format(100. * score))

    return score * 100.


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
