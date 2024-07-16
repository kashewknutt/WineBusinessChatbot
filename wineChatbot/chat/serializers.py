# chat/serializers.py
from rest_framework import serializers
from chat.models import CorpusEntry

class CorpusEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = CorpusEntry
        fields = ['question', 'answer']
