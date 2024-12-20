import os
import argparse
import json
import re

from askchart.eval.m4c_evaluator import Chart2tableAccuracyEvaluator
import ipdb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--COT', type=bool, default=False)
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token:' in prompt and len(prompt.split('\n')) >= 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file, COT=False):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    with open(annotation_file, 'r', encoding='utf-8') as file:
        annotations = [json.loads(line) for line in file]
    annotations = {(annotation['id'], annotation['text'].lower()): annotation for annotation in annotations} # return a dictionary with key as tuple (id, text) and value as annotation
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[(result['id'], result['prompt'].lower())]
        # print(f"Annotation: {annotation},id: {result['id']}, prompt after processor:{result['prompt'].lower()}")
        
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['label'],
        })
        # print(f"Pred_list: {pred_list}")
        # ipdb.set_trace()

    evaluator = Chart2tableAccuracyEvaluator()
    print(f'COT: {COT}')
    if COT:
        metrics = evaluator.eval_pred_list_COT(pred_list)
    else:
        metrics = evaluator.eval_pred_list(pred_list)
    print('Samples: {}\nMetrics: {}\n'.format(len(pred_list), metrics))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file, args.COT)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file), args.COT)
