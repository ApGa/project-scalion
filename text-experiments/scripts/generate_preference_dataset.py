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
                {"role":"system", "content": "Paraphrase the following question."},
                {"role":"user", "content":input_text}
                ])
            response_i_list.append([{"role": "assistant", "content":response_1}])
            response_j_list.append([{"role": "assistant", "content":response_2}])
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
        source_dataset = load_dataset("json", data_files=os.path.join(args.source_path, prompt, "output.jsonl"))["train"]

        for i in range(len(source_dataset)):
            original_query = source_dataset[i]['step'][0]['full_input'][-1]['content']
            paraphrased_list = source_dataset[i]['answer']

            for j in range(10):
                sample_id = 10*i+j
                line = responses[sample_id]
                res = line['step'][0]['full_input'][0]['content'].split("\\boxed{}.")[-1].strip()
                if res in paraphrased_list:
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
        # # source_dataset[0]["full_input"]
        # sample_id = 0
        # for i in range(0, len(responses), 10):
        #     print(responses[i:i+10])
        #     for line in [dict(zip(responses.column_names, values)) for values in responses[i:i+10]]:
        #         print(line)
        #         if sample_id in collected_responses:
        #             collected_responses[sample_id]["response"].extend(
        #                 line['step'][0]['completion']
        #             )
        #         else:
        #             collected_responses[sample_id] = {
        #                 "user_input": source_dataset[sample_id]['question'],
        #                 "ground_truth": line["ground_truth"],
        #                 "response": line['step'][0]['full_input'][-1]['content'],
        #                 "score": line["accuracy"],
        #             }
        #     sample_id += 1

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

    #df = pd.DataFrame(data)
    #df = df.sample(frac=1).reset_index(drop=True)
    #df.to_csv(os.path.join(args.output_path, "data.csv"), index=False)

