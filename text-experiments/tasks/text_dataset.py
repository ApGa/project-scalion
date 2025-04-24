import time
import os
import re
from functools import partial

from yeval.prompt import YevalPrompt
from yeval.task import register_task, YevalTask
from yeval.prompt import register_prompt, YevalPrompt

from yeval.task.commonsense_qa import CommonsenseQATask
from yeval.task.gsm8k import GSM8KTask
from yeval.response import extract_answer, get_boxed_answer

from yeval.metrics import math_eval

dir_path = os.path.dirname(os.path.realpath(__file__))

def extract_fn(answer: str):
    try:
        extracted_answer = answer.split('**Answer:**')[-1].strip()
        pattern = r'-?\d+(\.\d+)?'
        match = re.search(pattern, extracted_answer)
        if match:
            return match.group()
        else:
            return extracted_answer
    except Exception as e:
        return answer

def extract_boxed_fn(answer: str):
    try:
        extracted_answer = get_boxed_answer(answer)
        pattern = r'-?\d+(\.\d+)?'
        match = re.search(pattern, extracted_answer)
        if match:
            return match.group()
        else:
            return extracted_answer
    except Exception as e:
        return answer

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

@register_prompt("prompt_cot")
class CotPrompt(YevalPrompt):
    user_message="Reason step by step and put your final answer within \\boxed{}."
    postprocessor=extract_boxed_fn

@register_prompt("prompt_cot_answer")
class CotAnswerPrompt(YevalPrompt):
    user_message="Reason step by step and put your final answer after \"Answer:\"."
    postprocessor=extract_fn

@register_task("gsm_plus")
class GSMPLUSTask(YevalTask):
    data_path="qintongli/GSM-Plus"
    input_text=lambda x: x["question"]
    output_text=lambda x: x["answer"]
    preprocessing=lambda dataset: dataset.filter(lambda example: example["perturbation_type"] == "distraction insertion")
    test_split="test"
    evaluation={"accuracy": math_eval}

@register_task("test_gsm_plus")
class GSMPLUSTestTask(GSMPLUSTask):
    test_split="testmini"

@register_task("gsm_ic")
class GSMICTask(YevalTask):
    data_path="json"
    data_kwargs={"data_dir": os.path.join(dir_path, f"data/gsm_ic/")}
    input_text=lambda x: x["new_question"]
    output_text=lambda x: x["answer"]
    test_split="train"
    evaluation={"accuracy": math_eval}

@register_task("test_gsm_ic")
class GSMICTestTask(GSMICTask):
    test_split="test"

@register_task("gsm_symbolic")
class GSM8KSymbolicTask(YevalTask):
    data_path="json"
    data_kwargs={"data_dir": os.path.join(dir_path, f"data/gsm_symbolic/main/")}
    input_text=lambda x: x["question"]
    output_text=lambda x: x["answer"].split("####")[-1].strip()
    test_split="train"
    evaluation={"accuracy": math_eval}

@register_task("gsm_symbolic_p1")
class GSM8KSymbolicP1Task(GSM8KSymbolicTask):
    data_kwargs={"data_dir": os.path.join(dir_path, f"data/gsm_symbolic/p1/")}

@register_task("gsm_symbolic_p2")
class GSM8KSymbolicP2Task(GSM8KSymbolicTask):
    data_kwargs={"data_dir": os.path.join(dir_path, f"data/gsm_symbolic/p2/")}

@register_task("test_gsm_symbolic")
class GSM8KSymbolicTestTask(GSM8KSymbolicTask):
    test_split="test"

@register_task("test_gsm_symbolic_p1")
class GSM8KSymbolicP1TestTask(GSM8KSymbolicP1Task):
    test_split="test"

@register_task("test_gsm_symbolic_p2")
class GSM8KSymbolicP2TestTask(GSM8KSymbolicP2Task):
    test_split="test"

@register_task("generate_paraphrase")
class GSM8KParaphraseGenerate1(YevalTask):
    system_message="""\
Rewrite the question to make it as concise as possible. Remove unhelpful information. \
DO NOT provide the answer.\
"""
    # user_message=lambda x: f"{x}\nLet's paraphrase this\n"
    sampling_args={
        "n": 1,
        "temperature": 0.0,
        "stop": ["\n\n"],
        "extra_body":{"guided_regex": "Paraphrase:.*"}
        }

@register_task("generate_paraphrase_1")
class GSM8KParaphraseGenerate1(GSMPLUSTask):
    system_message="""\
Rewrite the question to make it as concise as possible. Remove unhelpful information. \
DO NOT provide the answer.\
"""
    # user_message=lambda x: f"{x}\nLet's paraphrase this\n"
    preprocessing=lambda dataset: partial(shuffle, seed=int(str(time.time()).split(".")[-1]))(dataset.filter(lambda example: example["perturbation_type"] == "distraction insertion"))
    # preprocessing=lambda x: partial(shuffle, seed=1000)(x)
    data_kwargs={"keep_in_memory": True}
    sampling_args={
        "n": 4,
        "temperature": 1.0,
        "stop": ["\n\n"],
        "extra_body":{"guided_regex": "Paraphrase:.*"}
        }
    
@register_task("generate_paraphrase_2")
class GSM8KParaphraseGenerate2(GSM8KParaphraseGenerate1):
    system_message="""\
Rewrite this so that it's as short as possible. Remove sentences that are irrelevant to the question. \
DO NOT provide the answer.\
"""
    # preprocessing=lambda x: partial(shuffle, seed=2000)(x)
    preprocessing=lambda dataset: partial(shuffle, seed=int(str(time.time()).split(".")[-1]))(dataset.filter(lambda example: example["perturbation_type"] == "distraction insertion"))
    
@register_task("generate_paraphrase_3")
class GSM8KParaphraseGenerate3(GSM8KParaphraseGenerate1):
    system_message="""\
Rewrite the question as concisely as possible. Only include information required to answer the question accurately. \
DO NOT provide the answer.\
"""
    # preprocessing=lambda x: partial(shuffle, seed=3000)(x)
    preprocessing=lambda dataset: partial(shuffle, seed=int(str(time.time()).split(".")[-1]))(dataset.filter(lambda example: example["perturbation_type"] == "distraction insertion"))
    
@register_task("generate_paraphrase_4")
class GSM8KParaphraseGenerate3(GSM8KParaphraseGenerate1):
    system_message="""You are a helpful rewriting model. \
Rewrite the question as concisely as possible. Only include information required to answer the question accurately. \
DO NOT provide the answer.\
"""
    # preprocessing=lambda x: partial(shuffle, seed=1004)(x)

@register_task("generate_paraphrase_with_feedback")
class GSM8KParaphraseGenerateWithFeedback(YevalTask):
    data_path="json"
    # data_kwargs={"data_dir": os.path.join(dir_path, f"data/gsm_symbolic/main/")}
    input_text=lambda x: x["input"]
    output_text=lambda x: x["ground_truth"]
    test_split="train"
    evaluation={"accuracy": math_eval}
    system_message="""You are a helpful question rewriting model. \
Your job is to paraphrase or reformat the question in a way that makes it easier for a solver to answer the question. \
"""
    preprocessing=lambda x: partial(shuffle, seed=1001)(x)
    sampling_args={
        "n": 5,
        "temperature": 1.0,
        "extra_body":{"guided_regex": "Paraphrase:.*"}
        }

def spread(dataset):

    def _spread(examples):
        all_idx = []
        all_sample_id = []
        all_sentence = []
        all_ground_truth = []
        all_original_string = []
        for idx, (sample_id, answer, ground_truth, original_string) in enumerate(
            zip(
                examples["sample_id"],
                examples["answer"],
                examples["ground_truth"],
                examples["step"],
                )
        ):
            answer = [ans.split("Paraphrase:")[-1].strip() for ans in answer]
            all_idx.extend([idx] * len(answer))
            all_sample_id.extend([sample_id] * len(answer))
            all_sentence.extend(answer)
            all_ground_truth.extend([str(ground_truth)] * len(answer))
            all_original_string.extend([original_string[0]["full_input"][-1]["content"]] * len(answer))

        return {
            "idx": all_idx,
            "original_string": all_original_string,
            "input": all_sentence,
            "output": all_ground_truth,
            }

    for key in dataset.num_columns.keys():
        dataset[key] = dataset[key].map(_spread, batched=True, remove_columns=dataset[key].column_names)
    return dataset

@register_task("score_paraphrase")
class ScoreParaphraseTask(YevalTask):
    user_message="Let's reason step by step and and then write the final answer within \\boxed{}."
    data_path="json"
    test_split="train"
    preprocessing=spread
    postprocessor=get_boxed_answer
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    evaluation={"accuracy": math_eval}
    aux_keys=["original_string"]
