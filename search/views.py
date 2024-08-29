from django.shortcuts import render
from .models import Document,Query,ProcessedDocument
from django.template import loader
from django.http import HttpResponse,JsonResponse
from django.views import View
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pickle
from nltk.corpus import words,stopwords
import enchant
import os
from django.conf import settings

def get_words_corpus():
    # Load bigrams_list and bigrams_freq from files
    valid_words_path = os.path.join(settings.BASE_DIR, 'data', 'valid_words.pkl')
    with open(valid_words_path, 'rb') as file:
        valid_words = pickle.load(file)
    return valid_words + words.words()

def get_bigrams_freq():
    bigrams_freq_path = os.path.join(settings.BASE_DIR, 'data', 'bigrams_freq.pkl')
    bigrams_list_path = os.path.join(settings.BASE_DIR, 'data', 'bigrams_list.pkl')
    
    with open(bigrams_freq_path, 'rb') as file:
        bigrams_freq = pickle.load(file)
    
    with open(bigrams_list_path, 'rb') as file:
        bigrams_list = pickle.load(file)
    
    return bigrams_list , bigrams_freq

valid_words = words.words()
words_corpus =  get_words_corpus()
bigrams_list,bigrams_freq = get_bigrams_freq()
english_dictionary = enchant.Dict("en_US")

# Initialize the FAISS index and load the model
dimension = 384  # Assume this dimension for the paraphrase-MiniLM-L6-v2 model
index = faiss.IndexFlatL2(dimension)
# model = SentenceTransformer('C:/Users/dell/Desktop/model/document_embeddings.pt')
model_path = os.path.join(settings.BASE_DIR, 'models', 'miniLM')
model = SentenceTransformer(model_path)
# model = SentenceTransformer('E:/Data_Analysis/models/miniLM')

# Load embeddings into FAISS index
processed_documents = ProcessedDocument.objects.all()
documents = Document.objects.all()
embeddings = np.array([doc.get_embedding() for doc in processed_documents])
# print(f'hii : {embeddings}')
index.add(embeddings)
# document_embeddings[idx]
# Create your views here.
def firstDocument(request):
      firstDoc = Document.objects.all().values()[1]
      template = loader.get_template('firstDoc.html')
      context = {
            'firstDoc': firstDoc,
        }
      return HttpResponse(template.render(context, request))

def searchResult(request):
      query_text = request.GET.get('query', '')  # Get the query from URL parameters
      if not query_text:
          query_text = 'Search Engine'

      query_text_after_processing = preprocess_text(query_text) 
      query_embedding = model.encode(query_text_after_processing, convert_to_tensor=True)
      query_embedding = np.array(query_embedding).reshape(1, -1)

      results = []
      formatted_match = ""

      exact_match = Document.objects.filter(document_text__icontains=query_text)[:1]
      print(exact_match)


      if exact_match.exists():
          for match in exact_match:
            formatted_match = add_new_lines(match.document_text)
            results.append(formatted_match)

      distances, indices = index.search(query_embedding, k=5)
      indices = [int(i) for i in indices[0]]
      for ind in indices:
          if documents.values()[ind]['document_text'] != formatted_match:     
            formatted_document = add_new_lines(documents.values()[ind]['document_text'])
            results.append(formatted_document)

      Query.objects.create(query_text=query_text, results_count=len(results))
      
      template = loader.get_template('search_result.html')

      languages = ["C++", "Python", "PHP", "Java", "C", "Ruby", 
                "R", "C#", "Dart", "Fortran", "Pascal", "Javascript"] 

      context = {
          'results': results,
          'languages' : languages,
        }
      return HttpResponse(template.render(context, request))

def add_new_lines(text):

    # Regular expression to match sentence-ending punctuation followed by a space and a capital letter
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)(?=\s[A-Z])\.'
    
    # Replace the match with the same match followed by a newline
    formatted_text = re.sub(pattern, r'\g<0>\n', text)

    return formatted_text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)

    # Remove short numbers
    text = re.sub(r'\b\d{1,2}\b', ' ', text)

    #Remove short words
    text = re.sub(r'\b\w{1,2}\b', ' ', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back into a single string
    return ' '.join(tokens)

def autocomplete_view(request):
    # print(f"this method is called with the query : {request.GET.get('query', '').strip().lower()}")
    query = request.GET.get('query', '').lower()
    suggestions = []

    if query:
        # Generate suggestions
        suggestions = autocomplete_with_bigrams_and_prefix(query, bigrams_freq, valid_words)
        print(f'suggestions are : {suggestions}')
    return JsonResponse({'suggestions': suggestions})

def autocorrect_view(request):
    query = request.GET.get('query', '').strip().lower()
    corrected_query = " "
    if query:
        corrected_query = check_query_for_autocorrection(query, valid_words,bigrams_freq,english_dictionary)
        return JsonResponse({'corrected_query': corrected_query})

def prefix_match(prefix,words_corpus):
    return [word for word in words_corpus if word.startswith(prefix)]

def bigram_suggestions(last_word,bigrams_freq):
    return [(w2.lower(),freq) for (w1,w2),freq in bigrams_freq.items() if w1 == last_word]

def autocomplete_with_bigrams_and_prefix(query, bigrams_freq, valid_words):
    words = query.split()
    last_word = words[-1] if words else ""
    print(last_word)
    # Get possible next words based on bigrams
    potential_next_words = bigram_suggestions(last_word, bigrams_freq)
    # Apply prefix matching to the potential next words
    prefix_matches = [w2 for w2, freq in potential_next_words if w2]
    # If there are no bigram matches, fall back to general prefix matching
    if not prefix_matches:
        prefix_matches = prefix_match(last_word, valid_words)
    
    # Sort the matches by frequency (in descending order)
    prefix_matches = sorted(prefix_matches, key=lambda w: bigrams_freq.get((last_word, w), 0), reverse=True)

    the_beggining_of_the_query = ' '.join(words)
    the_beggining_of_the_query = the_beggining_of_the_query.rsplit(' ', 1)[0]
    prefix_new = []
    print(f"the_beggining_of_the_query : {the_beggining_of_the_query}")
    for prefixmatch in prefix_matches[:6]:
        if prefixmatch in valid_words:
            prefix_new.append(the_beggining_of_the_query + ' ' + prefixmatch)

    return prefix_new[:5]

def check_query_for_autocorrection(query,valid_words,bigrams_freq,english_dictionary):
    stop_words_set = stopwords.words('english')
    splitted_query = query.split()
    possible_corrections = []
    corrected_qurey = ""
    corrected_qurey = corrected_qurey.join(query)
    misspelled_words = [word for word in splitted_query if word not in valid_words]
    if misspelled_words:
        for word in misspelled_words:
            # print(f'misspelled word is : {word}')
            previous_word_index = splitted_query.index(word) - 1
            previous_word = splitted_query[previous_word_index]
            possible_corrections = english_dictionary.suggest(word)
            if possible_corrections:
                if previous_word in stop_words_set:
                    try:
                        new_previous_word_index = splitted_query.index(word) - 2
                        autocorrected_word = autocorrect_word(splitted_query[new_previous_word_index],word,bigrams_freq,possible_corrections)
                        corrected_qurey = corrected_qurey.replace(word,autocorrected_word)
                    except:
                        continue
                else:
                        autocorrected_word = autocorrect_word(previous_word,word,bigrams_freq,possible_corrections)
                        corrected_qurey = corrected_qurey.replace(word,autocorrected_word)
    return corrected_qurey
  
def autocorrect_word(previous_word,word,bigrams_freq,possible_corrections):
    best_correction = None
    best_correction_freq = 0
    # print(possible_corrections)
    for correction in possible_corrections:
        bigram = (previous_word, correction.lower())
        freq = bigrams_freq.get(bigram, 0)
        if freq > best_correction_freq:
            best_correction_freq = freq
            best_correction = correction
    if best_correction:
            print(f'1) best correction for the word {word} is {best_correction}')
            return best_correction
    else:
           print(f'2) correction for the word {word} is {possible_corrections[0]}')
           return possible_corrections[0]
