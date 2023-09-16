import os
import PyPDF2
import textract
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import re

# Load the Hugging Face job descriptions dataset
job_descriptions = load_dataset("jacob-hugging-face/job-descriptions")

# Select 10-15 job descriptions
job_descriptions = job_descriptions['train']['job_description'][:15]

# Initialize DistilBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        text = textract.process(pdf_path).decode("utf-8")
    except Exception:
        text = ""
    return text

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Define a function to calculate similarity between CV and job descriptions
def calculate_similarity(cv_embedding, job_description_embedding):
    similarity_scores = cosine_similarity(cv_embedding.reshape(1, -1), job_description_embedding.reshape(1, -1))
    return similarity_scores[0][0]

# Specify the folder containing CVs in PDF format
cv_folder = "/content/data/data/ACCOUNTANT"  # Replace with the path to your CVs

# List to store the top 5 CVs for each job description
top_cv_matches = []

# Loop through each job description
for job_description in job_descriptions:
    job_description = preprocess_text(job_description)

    # Encode the job description
    job_description_encoding = tokenizer.encode(job_description, return_tensors="pt", truncation=True, padding='max_length', max_length=512)

    # Calculate embeddings for the job description
    job_description_embedding = model(job_description_encoding).last_hidden_state.mean(dim=1).detach().numpy()

    # Dictionary to store CVs and their similarity scores
    cv_similarities = {}

    # Loop through CVs in the folder
    for cv_file in os.listdir(cv_folder):
        cv_path = os.path.join(cv_folder, cv_file)
        cv_text = extract_text_from_pdf(cv_path)
        cv_text = preprocess_text(cv_text)

        # Encode the CV
        cv_encoding = tokenizer.encode(cv_text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)

        # Calculate embeddings for the CV
        cv_embedding = model(cv_encoding).last_hidden_state.mean(dim=1).detach().numpy()

        # Calculate cosine similarity
        similarity_scores = calculate_similarity(cv_embedding, job_description_embedding)

        # Store the CV and its similarity score
        cv_similarities[cv_file] = similarity_scores

    # Sort CVs by similarity score and select the top 5
    top_cv_matches.append(
        {
            "job_description": job_description,
            "top_cv_matches": dict(sorted(cv_similarities.items(), key=lambda item: item[1], reverse=True)[:5])
        }
    )

# Print the top 5 CVs for each job description
for i, job_matches in enumerate(top_cv_matches):
    print(f"Job Description {i + 1}:\n{job_matches['job_description']}\n")
    print("Top 5 CV Matches:")
    for cv_file, similarity in job_matches['top_cv_matches'].items():
        print(f"CV: {cv_file}, Similarity Score: {similarity:.4f}")
    print("\n")
