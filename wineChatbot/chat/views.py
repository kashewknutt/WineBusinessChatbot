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

@method_decorator(csrf_exempt, name='dispatch')
class ChatbotView(APIView):
    def post(self, request):
        question = request.data.get('question')
        conversation_history.append({'role': 'user', 'content': question})

        corpus = load_corpus()
        answer = get_best_match(question, corpus)
        if answer:
            response = answer
        else:
            response = "Please contact the business directly for more information."

        conversation_history.append({'role': 'bot', 'content': response})
        return Response({'answer': response})



def chat_view(request):
    return render(request, 'chat/chat.html')

