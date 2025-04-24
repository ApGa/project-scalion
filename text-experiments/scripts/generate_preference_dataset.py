import os
import argparse
import jsonlines
import multiprocessing

import pandas as pd

from tqdm import tqdm
from functools import partial
from datasets import load_dataset

def process_response(response):

    question_list = []
    response_i_list = []
    response_j_list = []
    gt_list = []

    input_text = response["user_input"]
    ground_truth = response["ground_truth"]
    response_dict = {
        "answers": response["response"],
        "score": response["score"],
    }

    all_responses = pd.DataFrame(response_dict)
    all_responses = all_responses.sort_values(by=['score'], ascending=False)
    all_responses = all_responses.reset_index(drop=True)

    for idx in range(len(all_responses)):
        preferred = all_responses.iloc[idx]
        if preferred['score'] == 0:
            break

        dispreferred = all_responses.iloc[idx+1:]
        if len(dispreferred) == 0:
            break

        response_1, score_1 = preferred
        for idx_2, (response_2, score_2) in dispreferred.iterrows():
            if score_2 == 1:
                continue
            question_list.append([
                {"role":"system", "content": "Rewrite the question to make it as concise as possible. Remove unhelpful information. DO NOT provide the answer."},
                {"role":"user", "content": input_text}
                ])
            response_i_list.append([{"role": "assistant", "content":"Paraphrase: "+response_1}])
            response_j_list.append([{"role": "assistant", "content":"Paraphrase: "+response_2}])
            gt_list.append(ground_truth)

    return question_list, response_i_list, response_j_list, gt_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--source_path', type=str)
    args = parser.parse_args()

    data = {
        'question':[],
        'response_i':[],
        'response_j':[],
        'ground_truth':[]
        }

    if os.path.exists(args.output_path) is False:
        os.makedirs(args.output_path)

    prompt_list = os.listdir(args.input_path)
    collected_responses = {}
    for prompt in prompt_list:

        responses = load_dataset("json", data_files=os.path.join(args.input_path, prompt, "output.jsonl"))['train']

        for line in responses:
            original_query = line["step"][0]["aux"]["original_string"]
            res = line['step'][0]['full_input'][0]['content'].split("\\boxed{}.")[-1].strip()
                
            if original_query in collected_responses:
                collected_responses[original_query]["response"].append(res)
                collected_responses[original_query]["score"].append(line["accuracy"])
            else:
                collected_responses[original_query] = {
                    "user_input": original_query,
                    "response": [res],
                    "ground_truth": line["ground_truth"],
                    "score": [line["accuracy"]],
                }

    result_list = list(tqdm(map(process_response, [response for _, response in collected_responses.items()]), total=len(collected_responses)))
    result_list = [result for result in result_list if len(result[0]) != 0]
    for (question_list, response_i_list, response_j_list, gt_list) in result_list:
        data['question'].extend(question_list)
        data['response_i'].extend(response_i_list)
        data['response_j'].extend(response_j_list)
        data['ground_truth'].extend(gt_list)

    os.makedirs(args.output_path, exist_ok=True)

    with jsonlines.open(os.path.join(args.output_path, "train.jsonl"), mode='w') as writer:
        for q, i, j, g in zip(data['question'], data['response_i'], data['response_j'], data['ground_truth']):
            line = {"ground_truth": g, "question": q, "response_i": i, "response_j": j}
            writer.write(line)

    test_samples = 128
    with jsonlines.open(os.path.join(args.output_path, "test.jsonl"), mode='w') as writer:
        for idx, (q, i, j, g) in enumerate(zip(data['question'], data['response_i'], data['response_j'], data['ground_truth'])):
            if idx == test_samples:
                break
            line = {"ground_truth": g, "question": q, "response_i": i, "response_j": j}
            writer.write(line)
