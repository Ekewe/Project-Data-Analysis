from text_analyzer import TopicModeling

def main():
    # Giving the path where to find the cleaned data
    file_path = 'C:/Users/Emman/Documents/GitHub/Project-Data-Analysis/data/raw/rows.json'
    pipeline = TopicModeling(file_path)

    # Define the number of topics
    n_topics = 50
    optimal_topics_result = pipeline.find_optimal_topics(max_topics=n_topics)

    # Visualize the BoW and TF-IDF top words as a wordcloud
    pipeline.vectorizers_to_wordcloud()

    # Visualize the coherence scores
    pipeline.visualize_scores(optimal_topics_result)

    # Visualize the words based on the coherence results in a wordcloud
    pipeline.wordcloud_visualization(pipeline.to_wordcloud_dict(optimal_topics_result)) # Here, it is first needed to convert the results to a structure the wordcloud function can read

    # Prints the results in the console
    pipeline.results_to_text(optimal_topics_result, n_topics)

if __name__ == '__main__':
    main()
