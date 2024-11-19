from text_analyzer import TopicModeling

def main():
    # Giving the path where to find the cleaned data
    file_path = 'C:/Users/Emman/Documents/GitHub/Project-Data-Analysis/data/raw/rows.json'
    pipeline = TopicModeling(file_path)

    # Define the number of topics to try
    n_topics = 50
    optimal_topics_result = pipeline.find_optimal_topics(max_topics=n_topics)
    print(f"Number of topics for the analysis: {n_topics}")

    # Visualize the coherence scores
    pipeline.visualize_scores(optimal_topics_result)

    # Print the results
    for result in optimal_topics_result:
        print(f"Vectorization/Modeling: {result['combo']}")
        print(f"Optimal Number of Topics: {result['Optimal Topics']}")
        print(f"Coherence Scores: {result['Optimal Scores']}")
        print("----------------------------------------")

if __name__ == '__main__':
    main()
