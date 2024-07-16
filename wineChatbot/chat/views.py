# chat/views.py
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from chat.models import CorpusEntry
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
import torch

conversation_history = []

def load_corpus():
    entries = CorpusEntry.objects.all()
    corpus = [{'question': entry.question, 'answer': entry.answer} for entry in entries]
    return corpus

def get_best_match(question, corpus):
    questions = [entry['question'] for entry in corpus]
    vectorizer = TfidfVectorizer().fit_transform([question] + questions)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_match_index = cosine_sim.argmax()
    if cosine_sim[best_match_index] > 0.5:  # You can adjust this threshold as needed
        return corpus[best_match_index]['answer']
    return None

class ChatbotView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = DistilBertForQuestionAnswering.from_pretrained('fine-tuned-model')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('fine-tuned-model')

    def post(self, request):
        question = request.data.get('question')
        conversation_history.append({'role': 'user', 'content': question})

        inputs = self.tokenizer(question, "Your context or additional information here", return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

        if not answer or answer == self.tokenizer.sep_token:
            response = "Please contact the business directly for more information."
        else:
            response = answer

        conversation_history.append({'role': 'bot', 'content': response})
        return Response({'answer': response})


def chat_view(request):
    return render(request, 'chat/chat.html')

