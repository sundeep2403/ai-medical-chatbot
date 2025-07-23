pip install pandas scikit-learn sentence-transformers
import pandas as pd

df = pd.read_csv("ai-medical-chatbot.csv", on_bad_lines='skip', engine='python')


df["full_question"] = df["Description"].fillna('') + " " + df["Patient"].fillna('')
questions = df["full_question"].tolist()
answers = df["Doctor"].fillna('').tolist()

from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')

question_embeddings = model.encode(questions, show_progress_bar=True)
