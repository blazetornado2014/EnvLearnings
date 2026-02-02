import numpy as np

def run_sft_simulation():
    print("=== FEB 4: SFT TUNING SIMULATOR ===\n")

    # 1. INITIAL STATE (The 'Pre-trained' weights)
    # Let's imagine our 'Apple' vector is currently at coordinate [0.8, 0.2]
    # where X = Tech-ness and Y = Fruit-ness.
    apple_vector = np.array([0.9, 0.1])
    
    # 2. THE TARGET (The 'Label' from our SFT dataset)
    # Our dataset says an Apple should be 100% Fruit [0.0, 1.0]
    target_vector = np.array([0.0, 1.0])
    
    learning_rate = 0.1
    print(f"Starting Vector: {apple_vector} (Mostly Tech)")
    print(f"Target Vector:   {target_vector} (All Fruit)\n")

    # 3. THE TRAINING LOOP (Backpropagation)
    for epoch in range(1, 11):
        # Calculate 'Loss' (How far are we from the target?)
        error = target_vector - apple_vector
        
        # Update weights: Move the vector a small step toward the target
        # Math: New_Vector = Old_Vector + (Learning_Rate * Error)
        apple_vector = apple_vector + (learning_rate * error)
        
        # Calculate 'Cosine Similarity' to see the progress
        similarity = np.dot(apple_vector, target_vector) / (np.linalg.norm(apple_vector) * np.linalg.norm(target_vector))
        
        print(f"Epoch {epoch:2}: Vector {apple_vector} | Similarity to Fruit: {similarity:.4f}")

    print("\nInsight: The vector 'drifted' toward Fruit while losing its Tech-ness.")

if __name__ == "__main__":
    run_sft_simulation()