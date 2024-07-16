import json
import os
from datasets import Dataset
from wineChatbot.settings import BASE_DIR

# Define the file path for the JSON corpus
file_path = os.path.join(BASE_DIR, 'chat/sampleQuestionAnswers.json')

# Load the JSON corpus
with open(file_path, 'r') as f:
    data = json.load(f)

# Convert to Hugging Face dataset format
qa_data = {
    'question': [],
    'context': [],
    'answers': []
}

# Define a detailed context for the chatbot
context = (
    "You are a highly knowledgeable and respectful chatbot specializing in wines. "
    "Your goal is to provide accurate and helpful information to users based on the available data about wines. "
    "Maintain a polite and ethical tone in all responses. If a question cannot be answered from the available data, "
    "politely suggest the user contact the business directly for more information."
)

# Populate the dataset with the corpus entries
for entry in data:
    qa_data['question'].append(entry['question'])
    qa_data['context'].append(context)
    qa_data['answers'].append({'text': [entry['answer']], 'answer_start': [0]})

# Create and save the Hugging Face dataset
dataset = Dataset.from_dict(qa_data)
dataset.save_to_disk(os.path.join(BASE_DIR, 'chat/qa_dataset'))

print("Dataset saved successfully!")
