from yeval.response.math_responses import get_boxed_answer
from yeval.prompt import YevalPrompt, register_prompt

class BaseBoxPrompt(YevalPrompt):
    postprocessor=get_boxed_answer

@register_prompt("reason0A")
class EngReasonBoxPrompt(BaseBoxPrompt):
    user_message="Reason step by step and put your final answer within \\boxed{}."

@register_prompt("reason0B")
class EngReasonBoxPrompt(BaseBoxPrompt):
    user_message=lambda x: f"{x}"+"\nReason step by step and put your final answer within \\boxed{}."

@register_prompt("reason1A")
class EngReasonBoxPrompt(BaseBoxPrompt):
    user_message="Think about it step by step and give your answer at the end in \\boxed{}."

@register_prompt("reason1B")
class EngReasonBoxPrompt(BaseBoxPrompt):
    user_message=lambda x: f"{x}"+"\nThink about it step by step and give your answer at the end in \\boxed{}."

@register_prompt("reason2A")
class EngReasonBoxPrompt(BaseBoxPrompt):
    user_message="First give step by step reasoning, then write the answer within \\boxed{}."

@register_prompt("reason2B")
class EngReasonBoxPrompt(BaseBoxPrompt):
    user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning, then write the answer within \\boxed{}."
