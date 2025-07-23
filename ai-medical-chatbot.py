pip install pandas scikit-learn sentence-transformers
import pandas as pd

df = pd.read_csv("ai-medical-chatbot.csv", on_bad_lines='skip', engine='python')


df["full_question"] = df["Description"].fillna('') + " " + df["Patient"].fillna('')
questions = df["full_question"].tolist()
answers = df["Doctor"].fillna('').tolist()

from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')

question_embeddings = model.encode(questions, show_progress_bar=True)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_response(user_input, top_k=1):
    user_embedding = model.encode([user_input])
    similarity_scores = cosine_similarity(user_embedding, question_embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    responses = []
    for idx in top_indices:
        responses.append({
            "matched_question": questions[idx],
            "response": answers[idx],
            "score": similarity_scores[idx]
        })
    return responses

print("Welcome to the AI Medical Chatbot. Ask your health question (type 'exit' to quit).\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = get_response(user_input)[0]
    print(f"\n🤖 Doctor: {response['response']}\n")
