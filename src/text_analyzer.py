import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import unicodedata
from gensim.models import LdaModel
from gensim.models import LsiModel
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel


class TopicModeling:
    def __init__(self, file_path, data=None, complaints=None, cleaned_complaints=None):
        """
        Initializes the TopicModeling class with the file path, NLP model, and vectorizers.

        Args:
            file_path (str): Path to the data file.

        Attributes:
            data (list or dict): Raw data loaded from the file.
            complaints (DataFrame): DataFrame containing the complaint texts.
            cleaned_complaints (DataFrame): DataFrame containing the cleaned and processed complaint texts.
            lemmatizer: Nltk WordNetLemmatizer.
            vectorizers (dict): Dictionary containing vectorization methods (BoW and TF-IDF).
            topic_models (dict): Dictionary containing topic modeling methods (LDA and LSA).
        """
        self.file_path = file_path
        self.data = data
        self.complaints = complaints
        self.cleaned_complaints = cleaned_complaints
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizers = {
            'BoW': self.bow_vectorizer,
            'TF-IDF': self.tfidf_vectorizer
        }
        self.topic_models = {
            'LDA': self.lda_model,
            'LSA': self.lsa_model
        }

        # Download once
        #nltk.download('punkt')
        #nltk.download('wordnet')
        #nltk.download('omw-1.4')

    def load_data(self):
        """
        Loads data from a JSON file and stores it in the `data` attribute.

        Returns:
            data (list or dict): The loaded data from the JSON file.
        """
        if not self.data:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        return self.data

    def extract_complaints(self):
        """
        Extracts the 'Complaint Text' data from the loaded JSON data and stores it in a DataFrame.

        Raises:
            KeyError: If the key 'data' is not found in the JSON structure.

        Returns:
            DataFrame: A DataFrame containing the extracted complaint texts.
        """
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
        """
        Cleans the extracted complaint texts by:
        - Converting text to lowercase.
        - Removing non-alphabetical characters and accents.
        - Removing stopwords.
        - Lemmatizing the text using spaCy.

        Returns:
            DataFrame: A DataFrame with cleaned complaint texts.
        """
        if not self.complaints:
            self.extract_complaints()
        # Load the english set of stopwords
        stop_words = set(stopwords.words('english'))
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].str.lower()
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].apply(lambda x: re.sub(r'[^a-zA-Z\sà-ÿÀ-ß]', '', x))
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn'))
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        # Apply lemmatization
        self.complaints['Complaint Text'] = self.complaints['Complaint Text'].apply(lambda x: ' '.join([self.lemmatizer.lemmatize(token) for token in word_tokenize(x)]))
        self.cleaned_complaints = self.complaints
        return self.cleaned_complaints

    @staticmethod
    def bow_vectorizer(data):
        """
        Vectorizes data using the Bag of Words (BoW) model.

        Args:
            data (list): List of cleaned complaint texts.

        Returns:
            Tuple (matrix, CountVectorizer): The BoW vectorized matrix and the vectorizer used.
        """
        vectorizer = CountVectorizer(max_features=500)
        return  vectorizer.fit_transform(data), vectorizer

    @staticmethod
    def tfidf_vectorizer(data):
        """
        Vectorizes data using the TF-IDF (Term Frequency-Inverse Document Frequency) model.

        Args:
            data (list): List of cleaned complaint texts.

        Returns:
            Tuple (matrix, TfidfVectorizer): The TF-IDF vectorized matrix and the vectorizer used.
        """
        vectorizer = TfidfVectorizer(max_features=500)
        return vectorizer.fit_transform(data), vectorizer

    @staticmethod
    def lda_model(corpus, dictionary, num_topics=50):
        """
        Performs Latent Dirichlet Allocation (LDA) topic modeling on pre-vectorized data.

        Args:
            corpus (list): The corpus created from the BoW or TF-IDF vectors.
            dictionary (Dictionary): The Gensim dictionary created from tokenized texts.
            num_topics (int): The number of topics to generate.

        Returns:
            LdaModel: The trained LDA model.
        """
        return LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=2)

    @staticmethod
    def lsa_model(corpus, dictionary, num_topics=50):
        """
        Performs Latent Semantic Analysis (LSA) topic modeling on pre-vectorized data.

        Args:
            corpus (list): The corpus created from the BoW or TF-IDF vectors.
            dictionary (Dictionary): The Gensim dictionary created from tokenized texts.
            num_topics (int): The number of topics to generate.

        Returns:
            LsiModel: The trained LSA model.
        """
        return LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

    @staticmethod
    def compute_coherence(model, texts, dictionary, coherence='c_v'):
        """
        Computes the coherence score for a given topic model.

        Args:
            model (Model): The topic model to evaluate.
            texts (list): The tokenized text data.
            dictionary (Dictionary): The Gensim dictionary used for creating the corpus.
            coherence (str): The type of coherence to calculate (default is 'c_v').

        Returns:
            float: The coherence score.
        """
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence)
        return coherence_model.get_coherence()

    def find_optimal_topics(self, max_topics=50, coherence='c_v'):
        """
        Finds the optimal number of topics for each vectorizer/model combination based on coherence scores.

        Args:
            max_topics (int): The maximum number of topics to test.
            coherence (str): The type of coherence to calculate (default is 'c_v').

        Returns:
            list: A list of dictionaries containing combo names, optimal topics, and coherence scores.
        """
        # Checks if the complaints are cleaned
        if self.cleaned_complaints is None or self.cleaned_complaints.empty:
            self.clean_data()

        cleaned_text = self.cleaned_complaints['Complaint Text']
        results = []

        for vec_name, vec_func in self.vectorizers.items():
            vectors, vectorizers = vec_func(cleaned_text)

            # Here we need to convert the vectors to a gensim structure
            id2word = {index: word for word, index in vectorizers.vocabulary_.items()}
            corpus = [[(word_id, count) for word_id, count in zip(doc.indices, doc.data)] for doc in csr_matrix(vectors)]
            tokenized_corpus = [[word for word in ({word: i for i, word in id2word.items()}).keys()]]
            gensim_dict = Dictionary(tokenized_corpus)

            for model_name, model_func in self.topic_models.items():
                # Results are stored into lists
                coherence_scores = []
                models = []
                for num_topics in range(1, max_topics + 1, 1):
                    model = model_func(corpus, gensim_dict, num_topics)
                    score = self.compute_coherence(model=model, texts=tokenized_corpus, dictionary=gensim_dict, coherence=coherence)
                    coherence_scores.append(score)
                    models.append(model)

                # Creation of variables to be called in Main
                optimal_score = max(coherence_scores)
                optimal_topics = coherence_scores.index(optimal_score) + 1
                optimal_model = models[coherence_scores.index(optimal_score)]
                optimal_model_topics = optimal_model.print_topics(num_topics=optimal_topics)

                # Append the results of the analysis to the lists
                results.append({
                    "combo": f"{vec_name}/{model_name}",
                    "Optimal Topics": optimal_topics,
                    "Optimal Scores": optimal_score,
                    "Topics": optimal_model_topics
                })

        return results

    @staticmethod
    def visualize_scores(results):
        """
        Visualizes the optimal coherence scores for each vectorization/model combination using a bar chart.

        Args:
            results (list): A list of dictionaries containing combo names and coherence scores.
        """
        combos = np.array([result['combo'] for result in results])
        scores =  np.array([result['Optimal Scores'] for result in results])

        # Sort results by coherence score
        sorted_indices = np.argsort(scores)
        combos = combos[sorted_indices]
        scores = scores[sorted_indices]

        # Plot coherence scores for each combination
        plt.figure(figsize=(12,6))
        plt.barh(combos, scores, color='skyblue')
        plt.xlabel('Coherence Score')
        plt.title('Optimal Coherence Scores')
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        # Display the graph
        plt.show()