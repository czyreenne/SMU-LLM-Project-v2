import json
from prediction_model import ModelPredictor
from system_prompts import sys_prompt
from datetime import datetime

def generate_answers(data, epoch):
    predictor = ModelPredictor(
        system_prompt=sys_prompt("ans_tort_qns"),
    )

    # Iterate over each file in the JSON data
    for item in data:
        scenario = item['scenario']
        questions = item['questions']
        answers = {}

        # For each question, prepare the prompt and get the model's prediction multiple times
        for i, question in enumerate(questions):
            question_key = f"question_{i + 1}"
            answers[question_key] = []

            for _ in range(epoch):
                prompt = f"Scenario: {scenario}\nQuestion: {question}\n\nMy answer is:"
                answer = predictor.predict(prompt)
                print(answer)
                answers[question_key].append(answer)

        # Add the answers to the data
        item['answers'] = answers

    return data

with open('data/processed/extracted_data.json', 'r') as file:
    data = json.load(file)

updated_data = generate_answers(data, epoch=3)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'results/exam_answers_{current_datetime}.json'
with open(output_filename, 'w') as file:
    json.dump(updated_data, file, indent=4)

print(f"Processing complete. The updated data is saved in '{output_filename}'.")
