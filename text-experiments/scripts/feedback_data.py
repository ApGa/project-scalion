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
    prev_paraphrase_questions: str = None,
) -> pd.DataFrame:
    def extract_prev_question(step):
        full_input = step[0]["full_input"]
        content = full_input[0]["content"]
        prev_question = "\n".join(content.split("\n")[1:].strip())
        return prev_question
    
    def format_question_and_answer(orig_question, answer, first_iteration: bool = False):
        if first_iteration:
            input_text = f"This is the original question provided to the solver: \"{orig_question}\".\n\The solver provided the following incorrect answer: {answer}.\n\n"
        input_text = f"This is a reworded version of the question provided to the solver: \"{orig_question}\".\n\The solver provided the following incorrect answer: {answer}.\n\n"
        return input_text
    
    def format_input_text(x, prev_model_paraphrase_df=None):
        question = x["orig_question"]
        answer = x["answer"]
        if prev_model_paraphrase_df is not None:
            prev_para_input = prev_model_paraphrase_df[prev_model_paraphrase_df["sample_id"] == x["sample_id"]]["orig_input"].values[0]
            formatted_qa = format_question_and_answer(question, answer)
            formatted_input = prev_para_input + formatted_qa
        else:
            formatted_input = format_question_and_answer(question, answer, first_iteration=True)
        return formatted_input
        
    results_df = pd.read_json(output_file, lines=True)
    # filter for incorrect examples
    incorrect = results_df[results_df["accuracy"] == 0.0]
    if len(incorrect.index) == 0:
        return None
    if prev_paraphrase_questions is not None:
        prev_model_paraphrase_df = pd.read_json(prev_paraphrase_questions, lines=True)
    else:
        prev_model_paraphrase_df = None
    feedback_data = pd.DataFrame()
    feedback_data["idx"] = incorrect["idx"]
    feedback_data["sample_id"] = incorrect["sample_id"]
    feedback_data["orig_question"] = incorrect["step"].apply(lambda x: extract_prev_question(x))
    feedback_data["answer"] = [x for x in incorrect["answer"].values]
    feedback_data["ground_truth"] = [x for x in incorrect["ground_truth"].values]
    feedback_data["input"] = feedback_data.apply(lambda x: format_input_text(x, args=prev_model_paraphrase_df), axis=1)
    return feedback_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output_filepath', type=str)
    parser.add_argument('--prev_paraphrase_questions', type=str)
    parser.add_argument('--save_filepath', type=str)
    args = parser.parse_args()
    feedback_data = create_input_text_with_feedback(args.model_output_filepath, args.prev_paraphrase_questions)
    if feedback_data is None:
        # no more incorrect examples to process
        sys.exit(2)
    else:
        with open(args.save_filepath, "w") as f: 
            f.write(feedback_data.to_json(orient='records', lines=True))
        
    