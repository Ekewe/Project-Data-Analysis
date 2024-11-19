import json
import re
from nltk.corpus import stopwords
import pandas as pd
import unicodedata
from gensim.models import LdaModel
from gensim.models import LsiModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
from gensim.corpora.dictionary import Dictionary
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel


class TopicModeling:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.complaints = None
        self.cleaned_complaints = None
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizers = {
            'BoW': self.bow_vectorizer,
            'TF-IDF': self.tfidf_vectorizer
        }
        self.topic_models = {
            'LDA': self.lda_model,
            'LSA': self.lsa_model
        }

    def load_data(self):
        """Loads data from JSON file."""
        if not self.data:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        return self.data

    def extract_complaints(self):
        """Extracts 'Complaints' data from the loaded JSON."""
        if not self.data:
            self.load_data()

        if 'data' in self.data:
            self.data = self.data['data']
        else:
            raise KeyError("Key 'data' not found in JSON file.")
        complaints_text = [entry[18] for entry in self.data if len(entry) > 18]
        self.complaints = pd.DataFrame({'Complaint Text': complaints_text})
        return self.complaints

    def clean_data(self):
        """Cleans the extracted 'Complaints' data."""
        if not self.complaints:
            self.extract_complaints()

        stop_words = set(stopwords.words('english'))

        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].str.lower()
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].apply(lambda x: re.sub(r'[^a-zA-Z\sà-ÿÀ-ß]', '', x))
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn'))
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        # Apply lemmatization
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].apply(lambda x: ' '.join([token.lemma_ for token in self.nlp(x)]))
        self.cleaned_complaints = self.complaints
        return self.cleaned_complaints

    def bow_vectorizer(self, data):
        """Vectorizes data using Bag of Words."""
        vectorizer = CountVectorizer(max_features=400)
        return  vectorizer.fit_transform(data), vectorizer

    def tfidf_vectorizer(self, data):
        """Vectorizes data using the TF-IDF."""
        vectorizer = TfidfVectorizer(max_features=400)
        return vectorizer.fit_transform(data), vectorizer

    def lda_model(self, corpus, dictionary, num_topics=50):
        """Performs LDA topic modeling."""
        return LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=5)

    def lsa_model(self, corpus, dictionary, num_topics=50):
        """Performs LSA topic modeling."""
        return LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

    def compute_coherence(self, model, data, vectorizer, coherence='c_v'):
        """Computes coherence score for the given model."""
        coherence_model = CoherenceModel(model=model, texts=data, dictionary=vectorizer, coherence=coherence)
        return coherence_model.get_coherence()

    def find_optimal_topics(self, max_topics=50, coherence='c_v'):
        """Finds the optimal number of topics for each vectorizer/model combination based on coherence score"""
        if self.cleaned_complaints is None or self.cleaned_complaints.empty:
            self.clean_data()

        cleaned_text = self.cleaned_complaints['Complaint Text']
        results = []

        for vec_name, vec_func in self.vectorizers.items():
            vectors, vectorizers = vec_func(cleaned_text)
            tokenized_texts = [doc.split() for doc in cleaned_text]
            dictionary = Dictionary(tokenized_texts)
            corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]
            for model_name, model_func in self.topic_models.items():
                coherence_scores = []

                for num_topics in range(1, max_topics+1, 3):
                    model = model_func(corpus, dictionary, num_topics)
                    score = self.compute_coherence(model, tokenized_texts, dictionary, coherence)
                    coherence_scores.append(score)

                optimal_score = max(coherence_scores)
                optimal_topics = coherence_scores.index(optimal_score) + 1
                results.append({
                    "combo": f"{vec_name}/{model_name}",
                    "Optimal Topics": optimal_topics,
                    "Coherence Scores": coherence_scores
                })

        return results

    def visualize_scores(self, results):
        """Visualizes the optimal coherence scores for each vectorization/model combination."""
        combos = [result['combo'] for result in results]
        scores = [result['optimal_coherence_score'] for result in results]

        # Plot coherence scores for each combination
        plt.figure(figsize=(12,6))
        plt.barh(combos, scores, color='skyblue')
        plt.xlabel('Coherence Score')
        plt.title('Optimal Coherence Scores')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        # Display the graph
        plt.show()