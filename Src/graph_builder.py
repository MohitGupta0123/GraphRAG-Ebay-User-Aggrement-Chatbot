
import os
import re
import fitz  # PyMuPDF
import contractions
from tqdm import tqdm
import nltk

nltk.download('punkt')
nltk.download('punkt-tab')

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    full_text = ""
    for page in tqdm(doc):
        full_text += page.get_text()
    doc.close()
    return full_text


def clean_text_debug(text):
    removed = {}
    text = text.replace("U.S.", "___US___")
    text = text.replace("U.K.", "___UK___")
    text = contractions.fix(text)
    text = text.replace("___US___", "U.S.")
    text = text.replace("___UK___", "U.K.")

    urls = re.findall(r'http\S+|www\S+|https\S+', text)
    removed['urls'] = urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    emails = re.findall(r'\S+@\S+', text)
    removed['emails'] = emails
    text = re.sub(r'\S+@\S+', '', text)

    text = re.sub(r'\n{2,}', '\n---\n', text)
    text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)

    pattern_to_keep = r"[^\w\s.,;:()?!\-/\'���\"$�]"
    non_alpha = re.findall(pattern_to_keep, text)
    removed['non_alpha'] = non_alpha
    text = re.sub(pattern_to_keep, '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text, removed


def save_cleaned_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Cleaned text saved to {output_path}")


def split_into_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)
