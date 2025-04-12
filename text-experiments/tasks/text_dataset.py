import os

from yeval.prompt import YevalPrompt
from yeval.task import register_task, YevalTask
from yeval.prompt import register_prompt, YevalPrompt

from yeval.task.commonsense_qa import CommonsenseQATask
from yeval.task.gsm8k import GSM8KTask
from yeval.response import extract_answer, get_boxed_answer

def extract_fn(answer: str):
    try:
        extracted_answer = answer.split('**Answer:**')[-1].strip()
        # take numbers only with regex
        return extracted_answer
    except Exception as e:
        return answer

@register_prompt("prompt_cot")
class CotPrompt(YevalPrompt):
    user_message="Reason step by step and put your final answer within \\boxed{}."
    postprocessor=get_boxed_answer

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

@register_task("gsm_symbolic_generate_paraphrase")
class GSM8KParaphraseTask(GSM8KSymbolicTask):
    system_message="""You are a helpful paraphrasing model. \
Write a paraphrase of the question. \
DO NOT provide the answer.\
"""
    sampling_args={"n": 10, "stop": ["Answer:"]}

dir_path = os.path.dirname(os.path.realpath(__file__))

@register_task("gsm_symbolic_paraphrased")
class GSM8KParaphraseTask(GSM8KSymbolicTask):
    data_path="json"
    data_kwargs={"data_files": {"test" : os.path.join(dir_path, "data/gsm_symbolic/output.jsonl")}}
    input_text=lambda x: x["answer"][0]
    output_text=lambda x: x["ground_truth"]
