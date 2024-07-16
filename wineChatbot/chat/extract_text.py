# extract_text.py
import os
import fitz
from wineChatbot.settings import BASE_DIR

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_path = os.path.join(BASE_DIR, 'chat/Corpus.pdf')
    corpus_text = extract_text_from_pdf(pdf_path)
    with open('corpus.txt', 'w') as file:
        file.write(corpus_text)
