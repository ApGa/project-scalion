import os
import re

from yeval.prompt import YevalPrompt
from yeval.task import register_task, YevalTask
from yeval.prompt import register_prompt, YevalPrompt

from yeval.task.commonsense_qa import CommonsenseQATask
from yeval.task.gsm8k import GSM8KTask
from yeval.response import extract_answer, get_boxed_answer

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

@register_prompt("prompt_cot")
class CotPrompt(YevalPrompt):
    user_message="Reason step by step and put your final answer within \\boxed{}."
    postprocessor=extract_boxed_fn

@register_prompt("prompt_cot_answer")
class CotAnswerPrompt(YevalPrompt):
    user_message="Reason step by step and put your final answer after \"Answer:\"."
    postprocessor=extract_fn

@register_task("gsm_symbolic")
class GSM8KSymbolicTask(YevalTask):
    data_path="apple/GSM-Symbolic"
    data_name="p1"
    input_text=lambda x: x["question"]
    output_text=lambda x: x["answer"].split("####")[-1].strip()
    test_split="test"
    evaluation={"accuracy": lambda x, y: x == y}

@register_task("gsm_symbolic_generate_paraphrase_1")
class GSM8KParaphraseGenerate1(GSM8KSymbolicTask):
    system_message="""You are a helpful paraphrasing model. \
Paraphrase or reformat the question in a way that makes it easier to answer. \
DO NOT provide the answer.\
"""
    sampling_args={"n": 10, "stop": ["Answer:"], "temperature": 1.0}
    
@register_task("gsm_symbolic_generate_paraphrase_2")
class GSM8KParaphraseGenerate2(GSM8KSymbolicTask):
    system_message="""You are a helpful rewriting model. \
Rewrite the question to make it easier to answer correctly. This rewrite can be a paraphrase or a formatting change. \
DO NOT provide the answer.\
"""
    sampling_args={"n": 10, "stop": ["Answer:"], "temperature": 1.0}
    
@register_task("gsm_symbolic_generate_paraphrase_3")
class GSM8KParaphraseGenerate3(GSM8KSymbolicTask):
    system_message="""You are a helpful reformatting model. \
Format the question in a way that makes it easiest to answer correctly. Only include the necessary information to answer the question correctly. \
DO NOT provide the answer.\
"""
    sampling_args={"n": 10, "stop": ["Answer:"], "temperature": 1.0}
    
@register_task("gsm_symbolic_generate_paraphrase_4")
class GSM8KParaphraseGenerate3(GSM8KSymbolicTask):
    system_message="""You are a helpful rewriting model. \
Rewrite the question as concisely as possible. Only include information required to answer the question accurately. \
DO NOT provide the answer.\
"""
    sampling_args={"n": 10, "stop": ["Answer:"], "temperature": 1.0}

dir_path = os.path.dirname(os.path.realpath(__file__))

def spread(dataset):

    def _spread(examples):
        all_sentence = []
        all_ground_truth = []
        all_idx = []
        for idx, answer, ground_truth in zip(examples["idx"], examples["answer"], examples["ground_truth"]):
            all_sentence.extend(answer)
            all_idx.extend([idx] * len(answer))
            all_ground_truth.extend([ground_truth] * len(answer))

        return {
            "idx": all_idx,
            "input": all_sentence,
            "output": all_ground_truth,
            }
    dataset["test"] = dataset["test"].map(_spread, batched=True, remove_columns=dataset["test"].column_names)
    return dataset

@register_task("gsm_symbolic_paraphrased")
class GSM8KParaphraseTask(GSM8KSymbolicTask):
    data_path="json"
    data_kwargs={"data_files": {"test" : os.path.join(dir_path, "data/gsm_symbolic_1/output.jsonl")}}
    preprocessing=spread
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
