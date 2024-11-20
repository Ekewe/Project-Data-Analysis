from text_analyzer import TopicModeling

def main():
    # Giving the path where to find the cleaned data
    file_path = 'C:/Users/Emman/Documents/GitHub/Project-Data-Analysis/data/raw/rows.json'
    pipeline = TopicModeling(file_path)

    # Define the number of topics
    n_topics = 20
    optimal_topics_result = pipeline.find_optimal_topics(max_topics=n_topics)

    # Visualize the coherence scores
    pipeline.visualize_scores(optimal_topics_result)

    # Print the results
    print(f"Number of topics for the analysis: {n_topics}")
    for result in optimal_topics_result:
        print(f"Vectorization/Modeling: {result['combo']}")
        print(f"Optimal Topic n°: {result['Optimal Topics']}")
        print(f"Coherence Scores: {result['Optimal Scores']}")
        print(f"Topics: {result['Topics']}")
        print("----------------------------------------")

if __name__ == '__main__':
    main()
