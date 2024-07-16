# chat/management/commands/load_sample_data.py
from django.core.management.base import BaseCommand
from chat.models import CorpusEntry
import json
import os
from wineChatbot.settings import BASE_DIR

class Command(BaseCommand):
    help = 'Load sample question-answer data'
    file_path = os.path.join(BASE_DIR, 'chat/sampleQuestionAnswers.json')

    def handle(self, *args, **kwargs):
        with open(Command.file_path, 'r') as file:
            data = json.load(file)
            for entry in data:
                CorpusEntry.objects.create(question=entry['question'], answer=entry['answer'])
        self.stdout.write(self.style.SUCCESS('Sample data loaded successfully.'))
