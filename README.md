# NLP Study Assistant Chatbot

A retrieval-based chatbot that answers beginner NLP and 
Semantic Web questions using TF-IDF vectorization and 
cosine similarity matching.

Built as a study tool while learning NLP concepts — 
the dataset covers core topics from tokenization and 
BERT to RDF, SPARQL, and Apache Jena.

## How it works

1. Loads a FAQ dataset from a CSV file (question + answer pairs)
2. Vectorizes all questions using TF-IDF
3. On user input, computes cosine similarity between 
   the query and all stored questions
4. Returns the answer for the best-matching question
5. Falls back to a default response if similarity is too low

## Why TF-IDF instead of embeddings?

TF-IDF is lightweight, interpretable, and works well for 
a small, well-defined FAQ dataset. For a larger or more 
open-ended knowledge base, sentence embeddings 
(e.g. Sentence Transformers) would give better semantic 
matching — which I explored in my RAG project.

## Example questions it can answer

- What is tokenization?
- What is BERT?
- What is SPARQL?
- What is sentiment analysis?
- What is Apache Jena?

## Tech stack

- Python
- pandas
- scikit-learn (TF-IDF, cosine similarity)

## Project structure

nlp-study-assistant-chatbot/
├── chatbot.py        # main logic
├── faq_data.csv      # question-answer dataset
├── requirements.txt
└── README.md

## Run locally

git clone https://github.com/tamarataha/NLP-study-assistant-chatbot
cd NLP-study-assistant-chatbot
pip install -r requirements.txt
python chatbot.py
