import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Load dataset
train_df = pd.read_csv("week-8\data\Training Dataset.csv")

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert rows to readable text documents
def row_to_text(row):
    return f"""Loan ID: {row['Loan_ID']}
Gender: {row['Gender']}, Married: {row['Married']}, Dependents: {row['Dependents']}
Education: {row['Education']}, Self Employed: {row['Self_Employed']}
Applicant Income: {row['ApplicantIncome']}, Coapplicant Income: {row['CoapplicantIncome']}
Loan Amount: {row['LoanAmount']}, Term: {row['Loan_Amount_Term']}, Credit History: {row['Credit_History']}
Property Area: {row['Property_Area']}
Loan Status: {row['Loan_Status']}"""

docs = train_df.apply(row_to_text, axis=1).tolist()
doc_embeddings = embedder.encode(docs, convert_to_numpy=True)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Load generator model (lightweight)
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [docs[i] for i in indices[0]]

def generate_answer(query):
    context = retrieve_context(query)
    context_text = "\n".join(context)
    prompt = f"Answer the question based on the context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=256, do_sample=True)
    return result[0]['generated_text']
