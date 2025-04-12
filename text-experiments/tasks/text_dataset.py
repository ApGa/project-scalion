import os

from yeval.prompt import YevalPrompt
from yeval.task import register_task, YevalTask

from yeval.task.commonsense_qa import CommonsenseQATask
from yeval.task.gsm8k import GSM8KTask
from yeval.response import extract_answer, get_boxed_answer

dir_path = os.path.dirname(os.path.realpath(__file__))

@register_task("commonsense_qa_generate_paraphrase")
class CommonsenseQAParaphraseTask(CommonsenseQATask):
    system_message="""You are a helpful paraphrasing model. \
Given a multiple choice question, write a paraphrase of the question-choices. \
DO NOT provide the answer.\
"""
    sampling_args={"n": 10, "stop": ["Answer:"]}

@register_task("gsm8k_generate_paraphrase")
class GSM8KParaphraseTask(GSM8KTask):
    system_message="""You are a helpful paraphrasing model. \
Given a multiple choice question, write a paraphrase of the question-choices. \
DO NOT provide the answer.\
"""
    sampling_args={"n": 10, "stop": ["Answer:"]}

@register_task("commonsense_qa_paraphrase")
class CommonsenseQAParaphraseTask(CommonsenseQATask):
    data_path="json"
    data_kwargs={"data_files": {"validation" : os.path.join(dir_path, "data/commonsense_qa/output.jsonl")}}
    input_text=lambda x: f"Answer with either A, B, C, D or E.\nQuestion:\n{x['answer'][0]}\nAnswer:"
    output_text=lambda x: x["ground_truth"]
    # sampling_args={"n": 10}

def gsm8k_paraphrased_input_text(x):
    input_text = x['answer'][0]
    input_text.split("Question:")[-1].strip()
    return f"Question:\n{input_text}\nAnswer:\nLet's think step by step.\n"

def gsm8k_input_text(x):
    return f"Question:\n{x['question']}\nAnswer:\nLet's think step by step.\n"

SYSYTEM_MESSAGE = """\
Let's think step by step and put the final answer in \\boxed{}.\
"""

@register_task("gsm8k_paraphrase")
class GSM8KParaphraseTask(GSM8KTask):
    data_path="json"
    data_kwargs={"data_files": {"validation" : os.path.join(dir_path, "data/gsm8k/output.jsonl")}}
    input_text=gsm8k_paraphrased_input_text
    output_text=lambda x: x["ground_truth"]
    system_message=SYSYTEM_MESSAGE
    postprocessor=get_boxed_answer
    # sampling_args={"n": 10}

@register_task("gsm8k_baseline")
class GSM8KBaseTask(GSM8KTask):
    input_text=gsm8k_input_text
    system_message=SYSYTEM_MESSAGE
    postprocessor=get_boxed_answer

