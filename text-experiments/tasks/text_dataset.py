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
MODEL = "Qwen/Qwen2.5-3B-Instruct"

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

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

@register_task("generate_paraphrase_1")
class GSM8KParaphraseGenerate1(GSM8KSymbolicTask):
    system_message="""You are a helpful paraphrasing model. \
Paraphrase or reformat the question in a way that makes it easier to answer. \
DO NOT provide the answer.\
"""
    preprocessing=lambda x: partial(shuffle, seed=1001)(x)
    sampling_args={
        "n": 10,
        "temperature": 1.0,
        "extra_body":{"guided_regex": "Paraphrase:.*"}
        }
    
@register_task("generate_paraphrase_2")
class GSM8KParaphraseGenerate2(GSM8KParaphraseGenerate1):
    system_message="""You are a helpful rewriting model. \
Rewrite the question to make it easier to answer correctly. This rewrite can be a paraphrase or a formatting change. \
DO NOT provide the answer.\
"""
    preprocessing=lambda x: partial(shuffle, seed=1002)(x)
    
@register_task("generate_paraphrase_3")
class GSM8KParaphraseGenerate3(GSM8KParaphraseGenerate1):
    system_message="""You are a helpful reformatting model. \
Format the question in a way that makes it easiest to answer correctly. Only include the necessary information to answer the question correctly. \
DO NOT provide the answer.\
"""
    preprocessing=lambda x: partial(shuffle, seed=1003)(x)
    
@register_task("generate_paraphrase_4")
class GSM8KParaphraseGenerate3(GSM8KParaphraseGenerate1):
    system_message="""You are a helpful rewriting model. \
Rewrite the question as concisely as possible. Only include information required to answer the question accurately. \
DO NOT provide the answer.\
"""
    preprocessing=lambda x: partial(shuffle, seed=1004)(x)

@register_task("generate_paraphrase_with_feedback")
class GSM8KParaphraseGenerateWithFeedback(YevalTask):
    data_path="json"
    # data_kwargs={"data_dir": os.path.join(dir_path, f"data/gsm_symbolic/main/")}
    input_text=lambda x: x["input"]
    output_text=lambda x: x["answer"].split("####")[-1].strip()
    test_split="train"
    evaluation={"accuracy": lambda x, y: x == y}
    system_message="""You are a helpful question rewriting model. \
Your job is to paraphrase or reformat the question in a way that makes it easier for a solver to answer the question. \
"""
    preprocessing=lambda x: partial(shuffle, seed=1001)(x)
    sampling_args={
        "n": 10,
        "temperature": 1.0,
        "extra_body":{"guided_regex": "Paraphrase:.*"}
        }

def spread(dataset):

    def _spread(examples):
        all_idx = []
        all_sample_id = []
        all_sentence = []
        all_ground_truth = []
        for idx, (sample_id, answer, ground_truth) in enumerate(zip(examples["sample_id"], examples["answer"], examples["ground_truth"])):
            answer = [ans.split("Paraphrase:")[-1].strip() for ans in answer]
            all_idx.extend([idx] * len(answer))
            all_sample_id.extend([sample_id] * len(answer))
            all_sentence.extend(answer)
            all_ground_truth.extend([ground_truth] * len(answer))

        return {
            "idx": all_idx,
            "sample_id": all_sample_id,
            "input": all_sentence,
            "output": all_ground_truth,
            }

    for key in dataset.num_columns.keys():
        dataset[key] = dataset[key].map(_spread, batched=True, remove_columns=dataset[key].column_names)
    return dataset

MODEL="Qwen/Qwen2.5-3B-Instruct"

@register_task("score_paraphrase")
class ScoreParaphraseTask(YevalTask):
    user_message="Let's reason step by step and and then write the final answer within \\boxed{}."
    data_path="json"
    test_split="train"
    # Set where the data is
    # data_kwargs={"data_files": {"test" : os.path.join(dir_path, f"data/paraphrased/{MODEL}/gsm_symbolic_1/output.jsonl")}}
    # data_kwargs={"data_files": os.path.join(dir_path, f"data/paraphrased/{MODEL}/gsm_symbolic_1/output.jsonl")}
    preprocessing=spread
    postprocessor=get_boxed_answer
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]

@register_task("gsm_symbolic_2_paraphrased")
class GSM8KParaphrase2Task(GSM8KParaphrase1Task):
    data_kwargs={"data_files": {"test" : os.path.join(dir_path, f"data/paraphrased/{MODEL.replace("/", "_")}/gsm_symbolic_2/output.jsonl")}}

@register_task("gsm_symbolic_3_paraphrased")
class GSM8KParaphrase3Task(GSM8KParaphrase1Task):
    data_kwargs={"data_files": {"test" : os.path.join(dir_path, f"data/paraphrased/{MODEL.replace("/", "_")}/gsm_symbolic_3/output.jsonl")}}

@register_task("gsm_symbolic_4_paraphrased")
class GSM8KParaphrase4Task(GSM8KParaphrase1Task):
    data_kwargs={"data_files": {"test" : os.path.join(dir_path, f"data/paraphrased/{MODEL.replace("/", "_")}/gsm_symbolic_4/output.jsonl")}}

@register_task("gsm_symbolic_paraphrased_with_feedback")
class GSM8KParaphrase1Task(GSM8KSymbolicTask):
    user_message="Let's reason step by step and and then write the final answer within \\boxed{}."
    data_path="json"
    test_split="test"
    preprocessing=spread
    postprocessor=get_boxed_answer
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]