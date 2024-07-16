# chat/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from chat.models import CorpusEntry
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

chatbot = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
conversation_history = []

def get_relevant_context(question, corpus):
    vectorizer = TfidfVectorizer().fit_transform([question] + [entry['question'] for entry in corpus])
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similar_docs = cosine_matrix[0][1:]
    sorted_indices = similar_docs.argsort()[-5:][::-1]  # Get top 5 relevant entries
    relevant_context = ' '.join([corpus[i]['answer'] for i in sorted_indices])
    return relevant_context, sorted_indices[0] if similar_docs[sorted_indices[0]] > 0.2 else None  # Threshold for relevance

@method_decorator(csrf_exempt, name='dispatch')
class ChatbotView(APIView):
    def post(self, request):
        question = request.data.get('question')
        conversation_history.append({'role': 'user', 'content': question})
        
        entries = CorpusEntry.objects.all()
        corpus = [{'question': entry.question, 'answer': entry.answer} for entry in entries]
        
        context, best_match_index = get_relevant_context(question, corpus)
        if best_match_index is not None:
            answer = corpus[best_match_index]['answer']
        else:
            answer = "Please contact the business directly for more information."
        
        conversation_history.append({'role': 'bot', 'content': answer})
        return Response({'answer': answer})


def chat_view(request):
    return render(request, 'chat/chat.html')

