"""
Script to filter examples that the solver LM gets incorrect
"""
import pandas as pd

def create_input_text_with_feedback(
    output_file: str,
):
    def extract_question(full_input):
        content = full_input[0]["content"]
        question = content.split("\n")[1].strip()
        return question
    
    def format_input_text(x):
        question = x["question"]
        answer = x["answer"]
        input_text = f"This is the original question provided to the solver: {question}. The solver incorrectly answered the question with the following answer: {answer}. Rephrase the question to make the problem easier to solve. Do not include the answer in your response."
        
    results_df = pd.read_json(output_file, lines=True)
    # filter for incorrect examples
    incorrect = results_df[results_df["accuracy"] == 0.0]
    feedback_data = pd.DataFrame()
    feedback_data["question"] = incorrect["full_input"].apply(lambda x: extract_question(x))
    feedback_data["answer"] = incorrect["answer"]
    feedback_data["input"] = feedback_data.apply(lambda x: format_input_text(x), axis=1)
    return feedback_data
    
    