# Topic Modeling with BoW and TF-IDF

This project implements **Topic Modeling** using **Bag-of-Words (BoW)** and **TF-IDF** vectorization techniques, with **Latent Dirichlet Allocation (LDA)** and **Latent Semantic Analysis (LSA)** models to discover topics in text data.

## Features

- **Text Cleaning**: Preprocesses the text (removes stopwords, lemmatizes, etc.).
- **Vectorization**: Uses **BoW** and **TF-IDF** for text representation.
- **Topic Modeling**: Applies **LDA** and **LSA** for topic discovery.
- **Coherence Scoring**: Computes coherence scores to evaluate the quality of the topics.
- **Visualization**: Visualizes coherence scores for different vectorization/model combinations.

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
1. Creation of an instance TopicModeling with the file path of the JSON data:
```python
pipeline = TopicModeling(file_path)
```
2. Clean the text: This step should be adjusted based on the source of your data (e.g., JSON, CSV). Ensure the correct data is extracted before applying the cleaning process.
```python
pipeline.clean_data()
```
3. You can then directly do the find_optimal_topics() with the max_topics parameters to choose how many topics should be passed:
```python
pipeline.find_optimal_topics(max_topics=50)
```
4. Visualize the results by passing it to the visualize_scores method:
```python
pipeline.visualize_scores(results)
```

## License
This project is licensed under the MIT License.