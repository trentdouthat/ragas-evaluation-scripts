import os
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# 1. LOAD CONFIG
load_dotenv()
print("Spinning up Embedding Model (No LLM required)...")

# 2. LOAD THE EMBEDDING MODEL
# This is the "Librarian" that turns text into numbers.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. DEFINE TEST DATA
question = "What command lists files in Linux?"
answer = "You use 'ls -l' to list files."
irrelevant_answer = "I really enjoy eating pizza with pineapple."

print(f"\nQuestion: {question}")
print(f"Good Answer: {answer}")
print(f"Bad Answer: {irrelevant_answer}")

# 4. CONVERT TEXT TO VECTORS (The "Dewey Decimal" Coordinates)
print("\nVectorizing text...")
q_vector = embeddings.embed_query(question)
a_vector = embeddings.embed_query(answer)
bad_a_vector = embeddings.embed_query(irrelevant_answer)

# 5. DEFINE THE MATH (Cosine Similarity)
def calculate_similarity(vec1, vec2):
    print(f"\n vec1: {vec1}")
    print(f"\n vec2: {vec2}")
    print(f"\n dot product: {np.dot(vec1, vec2)}")
    print(f"\n norm of vec1: {np.linalg.norm(vec1)}")
    print(f"\n norm of vec2: {np.linalg.norm(vec2)}")

    # Dot product of two vectors divided by the product of their lengths
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 6. RUN THE MATH
score_good = calculate_similarity(q_vector, a_vector)
score_bad = calculate_similarity(q_vector, bad_a_vector)

print("\n--- MANUAL EVALUATION RESULTS ---")
print(f"Similarity Score (Good Answer): {score_good:.4f}")  # Should be high (close to 1.0)
print(f"Similarity Score (Bad Answer):  {score_bad:.4f}")   # Should be low (close to 0.0)

if score_good > 0.5:
    print("\nSUCCESS: The pipeline correctly identified the relevant answer.")
else:
    print("\nFAIL: Something is wrong with the math.")