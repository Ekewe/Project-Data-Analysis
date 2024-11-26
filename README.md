# Topic Modeling with BoW and TF-IDF

This project implements **Topic Modeling** using **Bag-of-Words (BoW)** and **TF-IDF** vectorization techniques, with **Latent Dirichlet Allocation (LDA)** and **Latent Semantic Analysis (LSA)** models to discover topics in text data.

## Features

- **Text Cleaning**: Preprocesses the text data by converting it to lowercase, removing non-alphabetic characters, accents, and stopwords. A lemmatizer (NLTK) is used to treat variations of words (e.g., "running" and "ran") as the same root word.
- **Vectorization**: Uses BoW and TF-IDF for text representation.
    - BoW generates a basic representation of the frequency of words in the dataset.
    - TF-IDF adjusts word counts by considering how frequently a word appears in the entire corpus, giving more importance to terms that are unique to individual documents.
- **Topic Modeling**: Applies LDA and LSA to uncover topics in the data. Coherence scores are computed using the Gensim library to evaluate the quality of the topics.
    - LDA discovers the probabilistic distribution of topics across documents.
    - LSA reduces the dimensionality of the term-document matrix to reveal latent topics via Singular Value Decomposition (SVD).
- **Coherence Scoring**:  Computes coherence scores to evaluate how well the topics align with the documents.
- **Visualization**: Generates visualizations of the coherence scores and word clouds for the topics and vectorizers to offer both qualitative and quantitative insights.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ekewe/Project-Data-Analysis/
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage
1. Create an instance of the TopicModeling class with the file path to your JSON data:
```python
pipeline = TopicModeling(file_path)
```
2. Clean the text: This step should be adjusted based on your data source (e.g., JSON, CSV). Ensure that the correct data is extracted and cleaned before applying the analysis:
```python
pipeline.clean_data()
```
3. Find optimal topics: Run the topic modeling process with the desired number of topics. This function will optimize the number of topics based on coherence scores:
```python
results = pipeline.find_optimal_topics(max_topics=50)
```
4. Visualize the coherence scores: Display the coherence scores for the different vectorization/model combinations in a bar chart:
```python
pipeline.visualize_scores(results)
```
5. Visualize the word cloud: You can generate a word cloud from the results of the find_optimal_topics() function to visualize the most significant terms based on their weights:
```python
wordcloud_data = pipeline.to_wordcloud_dict(results)
pipeline.wordcloud_visualization(wordcloud_data)
```
6. Visualize the top words for BoW and TF-IDF vectorizers: This will show the most frequent words for both vectorization methods:
```python
pipeline.vectorizers_to_wordcloud(n_top_words=15)
```
7. Print the results: Print a summary of the optimal topics, coherence scores, and topics themselves:
```python
pipeline.results_to_text(results)
```

## Expected Outputs
Coherence scores: These are used to determine the quality of the topics discovered. Higher coherence indicates better alignment between topics and the documents.
Word clouds: Visual representations of the most important words in the topics, generated for both vectorization techniques and topics themselves.
Topic modeling results: LDA and LSA models, along with the topics they generate, are printed or visualized based on the results_to_text() function.

## Libraries Used
Gensim: For topic modeling and calculating coherence scores.
Scikit-learn: For the vectorization process (BoW and TF-IDF).
WordCloud: To generate word clouds for visualizing the most significant terms.
Matplotlib: For visualizations of coherence scores and topics.
Numpy: For numerical operations and handling arrays in the analysis.
Pandas: For data manipulation and analysis, especially for managing the complaints dataset.
Scipy: For handling sparse matrices, which is used during the vectorization process.


## License
This project is licensed under the MIT License.
