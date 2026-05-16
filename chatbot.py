import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_faq_data(file_path):
    return pd.read_csv(file_path)


def find_best_answer(user_question, faq_data):
    questions = faq_data["question"].tolist()
    all_questions = questions + [user_question]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(all_questions)

    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I do not know the answer to that yet. Try asking about NLP, BERT, RDF, or tokenization."

    return faq_data.iloc[best_match_index]["answer"]


def main():
    faq_data = load_faq_data("faq_data.csv")

    print("NLP Study Assistant Chatbot")
    print("Ask me a question. Type 'exit' to stop.\n")

    while True:
        user_question = input("You: ")

        if user_question.lower().strip() == "exit":
            print("Chatbot: Goodbye!")
            break

        answer = find_best_answer(user_question, faq_data)
        print("Chatbot:", answer)
        print()


if __name__ == "__main__":
    main()