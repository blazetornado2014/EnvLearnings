import numpy as np
from sentence_transformers import SentenceTransformer

def run_vector_lab():
    print("=== FEB 3: VECTOR SIMILARITY LAB ===\n")
    
    # Load a lightweight 'Map' (Model)
    # This model turns words into coordinates in a 384-dimension space.
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_similarity(word_a, word_b):
        # 1. Turn words into Vectors (Coordinates)
        v_a = model.encode([word_a])
        v_b = model.encode([word_b])
        
        # 2. Calculate Cosine Similarity (Math for 'How close are they pointing?')
        dot_product = np.dot(v_a, v_b.T)
        norm_a = np.linalg.norm(v_a)
        norm_b = np.linalg.norm(v_b)
        return (dot_product / (norm_a * norm_b))[0][0]

    # Test the relationships
    sim_fruit = get_similarity("Apple", "Fruit")
    sim_tech = get_similarity("Apple", "iPhone")
    sim_random = get_similarity("Apple", "Car")

    print(f"Similarity (Apple vs Fruit):  {sim_fruit:.4f}")
    print(f"Similarity (Apple vs iPhone): {sim_tech:.4f}")
    print(f"Similarity (Apple vs Car):    {sim_random:.4f}")
    
    print("\nInsight: The higher the number (closer to 1.0), the 'closer' they are in the AI's memory.")

if __name__ == "__main__":
    run_vector_lab()