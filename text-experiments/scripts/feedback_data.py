"""
Script to filter examples that the solver LM gets incorrect
and create data to feed to the paraphrasing model
"""
import argparse
import json
import pandas as pd
import sys

def create_input_text_with_feedback(
    output_file: str,
) -> pd.DataFrame:
    def extract_question(full_input):
        content = full_input[0]["content"]
        question = content.split("\n")[1].strip()
        return question
    
    def format_input_text(x):
        question = x["orig_question"]
        answer = x["answer"]
        input_text = f"This is the original question provided to the solver: {orig_question}. The solver incorrectly answered the question with the following answer: {answer}. Rephrase the question to make the problem easier to solve. Do not include the answer in your response."
        
    results_df = pd.read_json(output_file, lines=True)
    # filter for incorrect examples
    incorrect = results_df[results_df["accuracy"] == 0.0]
    if len(incorrect.index) == 0:
        return None
    feedback_data = pd.DataFrame()
    feedback_data["idx"] = incorrect["idx"]
    feedback_data["orig_question"] = incorrect["full_input"].apply(lambda x: extract_question(x))
    feedback_data["answer"] = incorrect["answer"]
    feedback_data["ground_truth"] = incorrect["ground_truth"]
    feedback_data["input"] = feedback_data.apply(lambda x: format_input_text(x), axis=1)
    return feedback_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output_filepath', type=str)
    parser.add_argument('--save_filepath', type=str)
    args = parser.parse_args()
    feedback_data = create_input_text_with_feedback(args.model_output_filepath)
    if feedback_data is None:
        # no more incorrect examples to process
        sys.exit(2)
    else:
        with open(args.save_filepath, "w") as f: 
            f.write(feedback_data.to_json(orient='records', lines=True))
        
    