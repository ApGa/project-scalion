"""Summarize strategies to update the search term."""

import os
import json
import litellm
import argparse

# %% Helper Functions

def extract_code_pieces(
    text: str, 
    start: str = "```", end: str = "```",
    do_split: bool = True,
) -> list[str]:
    """Extract code pieces from a text string.
    Args:
        text: str, model prediciton text.
    Rets:
        code_pieces: list[str], code pieces in the text.
    """
    code_pieces = []
    while start in text:
        st_idx = text.index(start) + len(start)
        if end in text[st_idx:]:
            end_idx = text.index(end, st_idx)
        else: 
            end_idx = len(text)
        
        if do_split:
            code_pieces.extend(text[st_idx:end_idx].strip().split("\n"))
        else:
            code_pieces.append(text[st_idx:end_idx].strip())
        text = text[end_idx+len(end):].strip()
    return code_pieces


def reformat_step(step: dict[str, str]) -> str:
    return f"{step['code']}  # {step['thought']}"


# %% Main Pipeline

def main():
    # load result
    task_id = args.result_dir.split("/")[-1].split('.')[1].split('_')[0]  # str, e.g., '250'
    instruction = json.load(open(os.path.join("config_files", f"{task_id}.json"), 'r'))["intent"]  # str
    trajectory = json.load(open(os.path.join(args.result_dir, "cleaned_steps.json"), 'r'))  # list[str]
    trajectory = [reformat_step(step) for step in trajectory]  # list[str]
    test_query = "##Test Example\nInstruction: " + instruction + "\n" + "Action Trajectory: \n```" + '\n'.join(trajectory) + "\n```\n\nSearch Strategy:"

    # summarize strategies
    messages = [{"role": "system", "content": open(args.sys_msg_path).read()}]
    messages += [{"role": "user", "content": open(args.instruction_path).read()}]
    messages += [{"role": "user", "content": test_query}]
    response = litellm.completion(
        api_key=os.environ.get("LITELLM_API_KEY"),
        base_url=os.environ.get("LITELLM_BASE_URL", "https://cmu.litellm.ai"),
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        n=args.num_responses,
    )["choices"][0]["message"]["content"]
    print(response)
    cont = input("Continue? (y/n): ")

    # save summary
    with open(args.output_path, 'a+') as f:
        f.write(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize strategies to update the search term.")
    # i/o path
    parser.add_argument("--result_dir", type=str, default="results/webarena.0", 
                        help="Directory containing the results.")
    parser.add_argument("--output_path", type=str, default="strategies.txt",
                        help="Path to save the summary of strategies.")
    
    # model
    parser.add_argument("--model", type=str, default="claude", choices=["gpt-4o", "claude"])
    parser.add_argument("--num_responses", type=int, default=1, help="Number of responses to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")

    # prompt
    parser.add_argument("--sys_msg_path", type=str, default="prompt/system_message.md")
    parser.add_argument("--instruction_path", type=str, default="prompt/instruction.md")

    args = parser.parse_args()

    if args.model == "claude":
        args.model = "litellm/neulab/claude-3-5-sonnet-20241022"
    args.model = args.model.replace("litellm", "openai")

    main()
